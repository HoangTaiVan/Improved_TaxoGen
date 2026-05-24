import os
import numpy as np
from collections import defaultdict, Counter

from .utils import (
    ensure_parent_dir,
    read_lines,
    write_lines,
)


# ======================================================
# FULL DATASET
# ======================================================

class DataSet:
    """
    Full dataset cho pipeline mới:
    SBERT + HDBSCAN + BM25/Cosine

    Input chính:
        papers.txt
        keywords.txt
        embeddings.txt

    embeddings.txt format:
        vocab_size dim
        keyword v1 v2 v3 ...
    """

    def __init__(
        self,
        document_file=None,
        keyword_file=None,
        embedding_file=None,
    ):
        self.document_file = document_file
        self.keyword_file = keyword_file
        self.embedding_file = embedding_file

        self.documents = []
        self.keywords = []

        self.keyword_set = set()
        self.keyword_to_id = {}

        # embeddings dạng dict:
        # {
        #   "keyword": np.array([...])
        # }
        self.embeddings = {}

        if document_file is not None:
            self.documents = self.load_documents(document_file)

        if keyword_file is not None:
            self.keywords = self.load_keywords(keyword_file)
            self.keyword_set = set(self.keywords)
            self.keyword_to_id = self.gen_keyword_id()

        if embedding_file is not None:
            self.embeddings = self.load_txt_embeddings(embedding_file)

            # Chỉ giữ keyword có embedding
            if self.keywords:
                self.keywords = [
                    kw for kw in self.keywords
                    if kw in self.embeddings
                ]
                self.keyword_set = set(self.keywords)
                self.keyword_to_id = self.gen_keyword_id()

    # ======================================================
    # LOADERS
    # ======================================================

    def load_documents(self, document_file):
        documents = []

        with open(document_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()

                if line:
                    documents.append(line.split())

        return documents

    def load_keywords(self, keyword_file):
        return read_lines(keyword_file)

    def gen_keyword_id(self):
        return {
            keyword: idx
            for idx, keyword in enumerate(self.keywords)
        }

    def load_txt_embeddings(self, embedding_file):
        """
        Load embeddings.txt kiểu Word2Vec text.

        Format:
            10000 384
            deep_learning 0.1 0.2 ...
            machine_translation 0.3 0.4 ...
        """

        if not embedding_file.endswith(".txt"):
            raise ValueError(
                "Embedding file phải là .txt, ví dụ: embeddings.txt hoặc phrase_embeddings.txt"
            )

        word_to_vec = {}

        with open(embedding_file, "r", encoding="utf-8") as fin:
            first_line = fin.readline().strip().split()

            has_header = (
                len(first_line) == 2
                and first_line[0].isdigit()
                and first_line[1].isdigit()
            )

            if not has_header and len(first_line) > 2:
                word = first_line[0]
                vec = np.array(
                    [float(x) for x in first_line[1:]],
                    dtype=np.float32,
                )
                word_to_vec[word] = vec

            for line in fin:
                items = line.strip().split()

                if len(items) < 3:
                    continue

                word = items[0]

                try:
                    vec = np.array(
                        [float(x) for x in items[1:]],
                        dtype=np.float32,
                    )
                except ValueError:
                    continue

                word_to_vec[word] = vec

        print(f"[LOAD] embeddings.txt keywords: {len(word_to_vec)}")

        return word_to_vec

    # ======================================================
    # OUTPUT FILES
    # ======================================================

    def save_doc_ids(self, output_file):
        lines = [str(i) for i in range(len(self.documents))]
        write_lines(lines, output_file)

    def build_keyword_cnt_file(self, output_file):
        """
        Sinh keyword_cnt.txt

        Format:
            doc_id<TAB>keyword<TAB>count<TAB>keyword<TAB>count...
        """

        ensure_parent_dir(output_file)

        with open(output_file, "w", encoding="utf-8") as fout:

            for doc_id, doc in enumerate(self.documents):

                counter = Counter(doc)

                row = [str(doc_id)]

                for kw, cnt in counter.items():

                    if kw not in self.keyword_set:
                        continue

                    row.append(kw)
                    row.append(str(cnt))

                fout.write("\t".join(row) + "\n")

                if (doc_id + 1) % 10000 == 0:
                    print(f"[keyword_cnt] processed {doc_id + 1} docs")

        print(f"[SAVE] keyword_cnt.txt: {output_file}")

    def build_index_file(self, output_file):
        """
        Sinh index.txt

        Format:
            keyword<TAB>doc_id1,doc_id2,...
        """

        ensure_parent_dir(output_file)

        phrase_docs = {
            kw: []
            for kw in self.keywords
        }

        keyword_set = set(self.keywords)

        for doc_id, doc in enumerate(self.documents):

            doc_set = set(doc)

            for kw in doc_set:
                if kw in keyword_set:
                    phrase_docs[kw].append(str(doc_id))

            if (doc_id + 1) % 10000 == 0:
                print(f"[index] processed {doc_id + 1} docs")

        with open(output_file, "w", encoding="utf-8") as fout:

            for kw in self.keywords:
                doc_ids = phrase_docs.get(kw, [])
                fout.write(f"{kw}\t{','.join(doc_ids)}\n")

        print(f"[SAVE] index.txt: {output_file}")


# ======================================================
# SUB DATASET
# ======================================================

class SubDataSet:
    """
    Dataset cho từng node trong taxonomy.

    Điểm quan trọng:
        Node dùng embedding riêng của node nếu truyền node_embedding_file.

    Nếu không truyền node_embedding_file:
        fallback về full_data.embeddings.
    """

    def __init__(
        self,
        full_data,
        doc_id_file,
        keyword_file,
        node_embedding_file=None,
    ):
        self.full_data = full_data
        self.node_embedding_file = node_embedding_file

        self.doc_ids = self.load_doc_ids(doc_id_file)

        if node_embedding_file is not None:
            self.node_embeddings_map = self.load_txt_embeddings(node_embedding_file)
        else:
            self.node_embeddings_map = self.full_data.embeddings

        self.keywords = self.load_keywords(keyword_file)

        self.keyword_set = set(self.keywords)

        self.keyword_to_id = self.gen_keyword_id()

        self.documents, self.original_doc_ids = self.load_documents()

        self.embeddings = self.load_embeddings()

        self.keyword_idf = self.build_keyword_idf()

    # ======================================================
    # LOAD
    # ======================================================

    def load_doc_ids(self, doc_id_file):

        doc_ids = []

        with open(doc_id_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()

                if line:
                    doc_ids.append(int(line))

        return doc_ids

    def load_txt_embeddings(self, embedding_file):
        word_to_vec = {}

        with open(embedding_file, "r", encoding="utf-8") as fin:
            first_line = fin.readline().strip().split()

            has_header = (
                len(first_line) == 2
                and first_line[0].isdigit()
                and first_line[1].isdigit()
            )

            if not has_header and len(first_line) > 2:
                word = first_line[0]
                vec = np.array(
                    [float(x) for x in first_line[1:]],
                    dtype=np.float32,
                )
                word_to_vec[word] = vec

            for line in fin:
                items = line.strip().split()

                if len(items) < 3:
                    continue

                word = items[0]

                try:
                    vec = np.array(
                        [float(x) for x in items[1:]],
                        dtype=np.float32,
                    )
                except ValueError:
                    continue

                word_to_vec[word] = vec

        return word_to_vec

    def load_keywords(self, keyword_file):

        keywords = []

        with open(keyword_file, "r", encoding="utf-8") as fin:

            for line in fin:

                keyword = line.strip()

                if not keyword:
                    continue

                if keyword in self.node_embeddings_map:
                    keywords.append(keyword)

                else:
                    print(f"[WARN] {keyword} not found in node embeddings")

        return keywords

    def gen_keyword_id(self):

        return {
            keyword: idx
            for idx, keyword in enumerate(self.keywords)
        }

    def load_embeddings(self):

        vectors = []

        for keyword in self.keywords:

            if keyword not in self.node_embeddings_map:
                continue

            vectors.append(
                self.node_embeddings_map[keyword]
            )

        return np.array(vectors, dtype=np.float32)

    def load_documents(self):

        trimmed_doc_ids = []
        trimmed_docs = []

        for doc_id in self.doc_ids:

            if doc_id < 0 or doc_id >= len(self.full_data.documents):
                continue

            doc = self.full_data.documents[doc_id]

            trimmed_doc = [
                token
                for token in doc
                if token in self.keyword_set
            ]

            if len(trimmed_doc) > 0:
                trimmed_doc_ids.append(doc_id)
                trimmed_docs.append(trimmed_doc)

        return trimmed_docs, trimmed_doc_ids

    # ======================================================
    # TF-IDF
    # ======================================================

    def build_keyword_idf(self):

        keyword_idf = defaultdict(float)

        for doc in self.documents:

            word_set = set(doc)

            for word in word_set:
                keyword_idf[word] += 1.0

        N = len(self.documents)

        if N == 0:
            return keyword_idf

        for word in keyword_idf:
            keyword_idf[word] = np.log(
                1.0 + N / keyword_idf[word]
            )

        return keyword_idf

    # ======================================================
    # CLUSTER OUTPUT
    # ======================================================

    def write_cluster_members(
        self,
        labels,
        cluster_file,
        parent_dir,
    ):
        """
        cluster_keywords.txt

        Format:
            cluster_id<TAB>keyword
        """

        ensure_parent_dir(cluster_file)

        cluster_map = defaultdict(list)

        with open(cluster_file, "w", encoding="utf-8") as fout:

            for keyword, label in zip(self.keywords, labels):

                label = int(label)

                if label == -1:
                    continue

                cluster_map[label].append(keyword)

                fout.write(f"{label}\t{keyword}\n")

        for label, members in cluster_map.items():

            child_dir = os.path.join(
                parent_dir,
                f"cluster_{label}"
            )

            os.makedirs(child_dir, exist_ok=True)

            seed_file = os.path.join(
                child_dir,
                "seed_keywords.txt"
            )

            write_lines(members, seed_file)

        return cluster_map

    # ======================================================
    # DOCUMENT MEMBERSHIP
    # ======================================================

    def get_doc_membership(self, labels):

        valid_labels = sorted(
            [int(x) for x in set(labels) if int(x) != -1]
        )

        if len(valid_labels) == 0:
            return {}

        keyword_cluster = {}

        for kw, label in zip(self.keywords, labels):

            label = int(label)

            if label == -1:
                continue

            keyword_cluster[kw] = label

        doc_cluster_map = {}

        for doc_id, doc in zip(
            self.original_doc_ids,
            self.documents
        ):

            scores = {
                label: 0.0
                for label in valid_labels
            }

            for kw in doc:

                if kw not in keyword_cluster:
                    continue

                cluster_id = keyword_cluster[kw]

                scores[cluster_id] += self.keyword_idf.get(kw, 0.0)

            best_cluster = max(
                scores,
                key=scores.get
            )

            if scores[best_cluster] > 0:
                doc_cluster_map[doc_id] = best_cluster

        return doc_cluster_map

    def write_document_membership(
        self,
        labels,
        output_file,
        parent_dir,
    ):
        """
        paper_cluster.txt

        Format:
            doc_id<TAB>cluster_id
        """

        ensure_parent_dir(output_file)

        cluster_doc_map = defaultdict(list)

        doc_cluster_map = self.get_doc_membership(labels)

        with open(output_file, "w", encoding="utf-8") as fout:

            for doc_id, cluster_id in doc_cluster_map.items():

                cluster_doc_map[cluster_id].append(doc_id)

                fout.write(f"{doc_id}\t{cluster_id}\n")

        for cluster_id, doc_ids in cluster_doc_map.items():

            child_dir = os.path.join(
                parent_dir,
                f"cluster_{cluster_id}"
            )

            os.makedirs(child_dir, exist_ok=True)

            doc_id_file = os.path.join(
                child_dir,
                "doc_ids.txt"
            )

            write_lines(doc_ids, doc_id_file)

        return cluster_doc_map