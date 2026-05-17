import json
import random
import copy


class TNode:

    def __init__(self, name, ph_list=None):

        self.name = name

        if ph_list is None:
            ph_list = []

        self.ph_list = ph_list
        self.ph_cnt = len(ph_list)

        self.parent = None
        self.children = []

        self.level = len(name.split("/"))

    def add_child(self, node):

        if node not in self.children:
            self.children.append(node)
            node.parent = self

    def get_siblings(self):

        if self.parent is None:
            return []

        sibs = copy.copy(self.parent.children)

        if self in sibs:
            sibs.remove(self)

        return sibs

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class Taxonomy:

    def __init__(self):

        self.root = TNode("*", [])

        self.all_nodes = {
            self.root.name: self.root
        }

    # ======================================================
    # ADD NODE
    # ======================================================

    def add_node(self, node):

        if node.name in self.all_nodes:
            return

        self.all_nodes[node.name] = node

        parts = node.name.split("/")

        if len(parts) == 1:
            parent_name = "*"
        else:
            parent_name = "/".join(parts[:-1])

        parent_node = self.find_node(parent_name)

        if parent_node is not None:
            parent_node.add_child(node)

    # ======================================================
    # FIND NODE
    # ======================================================

    def find_node(self, name):

        if name in self.all_nodes:
            return self.all_nodes[name]

        return None

    # ======================================================
    # RANDOM SAMPLE
    # ======================================================

    def sample_a_node(self):

        node_names = list(self.all_nodes.keys())

        if "*" in node_names:
            node_names.remove("*")

        if len(node_names) == 0:
            return self.root

        rand_name = random.choice(node_names)

        return self.all_nodes[rand_name]

    # ======================================================
    # EXPORT TXT
    # ======================================================

    def export_txt(self, output_file):

        with open(output_file, "w", encoding="utf-8") as f:

            sorted_nodes = sorted(
                self.all_nodes.values(),
                key=lambda x: x.name
            )

            for node in sorted_nodes:

                if node.name == "*":
                    continue

                keywords = ",".join(node.ph_list)

                f.write(
                    f"{node.name}\t{keywords}\n"
                )

        print(f"[SAVE] taxonomy txt: {output_file}")

    # ======================================================
    # EXPORT JSON
    # ======================================================

    def export_json(self, output_file):

        def build_json(node):

            return {
                "name": node.name,
                "keywords": node.ph_list,
                "children": [
                    build_json(child)
                    for child in sorted(
                        node.children,
                        key=lambda x: x.name
                    )
                ]
            }

        taxonomy_json = build_json(self.root)

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                taxonomy_json,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[SAVE] taxonomy json: {output_file}")

    # ======================================================
    # PRINT TREE
    # ======================================================

    def print_tree(self):

        def recur(node, depth=0):

            indent = "  " * depth

            if node.name == "*":
                print("* ROOT")
            else:
                kws = ", ".join(node.ph_list[:5])

                print(
                    f"{indent}- {node.name} "
                    f"[{kws}]"
                )

            for child in sorted(
                node.children,
                key=lambda x: x.name
            ):
                recur(child, depth + 1)

        recur(self.root)

    # ======================================================
    # GET ALL LEAF NODES
    # ======================================================

    def get_leaf_nodes(self):

        return [
            node
            for node in self.all_nodes.values()
            if node.is_leaf() and node.name != "*"
        ]

    # ======================================================
    # GET INTERNAL NODES
    # ======================================================

    def get_internal_nodes(self):

        return [
            node
            for node in self.all_nodes.values()
            if not node.is_leaf() and node.name != "*"
        ]

    # ======================================================
    # STATS
    # ======================================================

    def statistics(self):

        total_nodes = len(self.all_nodes) - 1

        leaf_nodes = len(self.get_leaf_nodes())

        internal_nodes = len(self.get_internal_nodes())

        max_depth = 0

        for node in self.all_nodes.values():

            if node.name == "*":
                continue

            depth = len(node.name.split("/"))

            if depth > max_depth:
                max_depth = depth

        return {
            "total_nodes": total_nodes,
            "leaf_nodes": leaf_nodes,
            "internal_nodes": internal_nodes,
            "max_depth": max_depth,
        }