import argparse
import random
import os
import sys

from os import listdir, makedirs
from os.path import isfile, join, basename, exists

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(ROOT_DIR)

from common.taxonomy import Taxonomy, TNode


# ======================================================
# READ TAXONOMY
# ======================================================

def read_taxonomy(tax_f):
    root = TNode('*', [])

    tax = Taxonomy()
    tax.root = root
    tax.all_nodes = {'*': root}

    with open(tax_f, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\r\n')

            if not line:
                continue

            if '\t' not in line:
                continue

            node_name, ph_str = line.split('\t', 1)

            if node_name == '*':
                continue

            ph_list = [
                p.strip()
                for p in ph_str.split(',')
                if p.strip()
            ]

            if len(ph_list) == 0:
                continue

            node = TNode(node_name, ph_list)
            tax.add_node(node)

    return tax


# ======================================================
# TOPIC INTRUSION
# ======================================================

def gen_intrusion_pairs(tax, N, case_N):
    cnt = 0
    pairs = {}

    max_attempts = case_N * 1000
    attempts = 0

    while cnt < case_N and attempts < max_attempts:
        attempts += 1

        node = tax.sample_a_node()

        if node.parent is None:
            continue

        siblings = node.get_siblings()

        if len(siblings) == 0:
            continue

        if len(node.ph_list) < N:
            continue

        sibling = random.choice(siblings)

        if len(sibling.ph_list) == 0:
            continue

        shuf_phs = list(node.ph_list[:N])

        cand_phs = [
            ph for ph in sibling.ph_list
            if ph not in shuf_phs
        ]

        if len(cand_phs) == 0:
            continue

        intruder = random.choice(cand_phs)

        shuf_phs.append(intruder)
        random.shuffle(shuf_phs)

        exp_line = ','.join(shuf_phs)
        intr_idx = shuf_phs.index(intruder)

        if exp_line not in pairs:
            pairs[exp_line] = intr_idx
            cnt += 1

    if cnt < case_N:
        print(f'[WARN] gen_intrusion_pairs: chỉ tạo được {cnt}/{case_N} pairs')

    return pairs


# ======================================================
# PARENT-CHILD / SUBDOMAIN Y-N TASK
# ======================================================

def gen_subdomain_pairs(tax, isa_N, case_N):
    """
    Sinh parent-child pairs dạng:

        child_terms,parent_terms

    Người đánh giá sẽ điền:
        y / n

    Khác bản cũ:
        - không còn 1 node = 1 pair
        - random sample nhiều lần để lấy đủ case_N
        - cùng một node có thể sinh nhiều biến thể keyword khác nhau
    """

    cnt = 0
    pairs = {}

    max_attempts = case_N * 1000
    attempts = 0

    while cnt < case_N and attempts < max_attempts:
        attempts += 1

        node = tax.sample_a_node()

        if node.name == '*':
            continue

        if node.parent is None:
            continue

        # bỏ quan hệ với root vì root không phải topic thật
        if node.parent.name == '*':
            continue

        if len(node.ph_list) == 0:
            continue

        if len(node.parent.ph_list) == 0:
            continue

        child_terms = random.sample(
            node.ph_list,
            min(isa_N, len(node.ph_list))
        )

        parent_terms = random.sample(
            node.parent.ph_list,
            min(isa_N, len(node.parent.ph_list))
        )

        if len(child_terms) == 0 or len(parent_terms) == 0:
            continue

        child_phs = '|'.join(child_terms)
        parent_phs = '|'.join(parent_terms)

        exp_line = f'{child_phs},{parent_phs}'

        if exp_line not in pairs:
            pairs[exp_line] = node.level
            cnt += 1

    if cnt < case_N:
        print(f'[WARN] gen_subdomain_pairs: chỉ tạo được {cnt}/{case_N} pairs')

    return pairs


# ======================================================
# OPTIONAL ISA TASK
# ======================================================

def gen_isa_pairs(tax, isa_N, case_N):
    """
    Task chọn parent đúng trong 2 lựa chọn.
    Hiện chưa ghi ra CSV trong bản này,
    nhưng giữ lại nếu sau này muốn dùng paper-style ISA.
    """

    cnt = 0
    pairs = {}

    max_attempts = case_N * 1000
    attempts = 0

    while cnt < case_N and attempts < max_attempts:
        attempts += 1

        node = tax.sample_a_node()
        rmd_node = tax.sample_a_node()

        if node.parent is None:
            continue

        if node.parent.name == '*':
            continue

        if rmd_node.name == '*':
            continue

        if node.parent == rmd_node:
            continue

        p_node = node.parent

        if len(node.ph_list) == 0:
            continue

        if len(p_node.ph_list) == 0:
            continue

        if len(rmd_node.ph_list) == 0:
            continue

        n_phs = '|'.join(node.ph_list[:isa_N])
        p_phs = '|'.join(p_node.ph_list[:isa_N])
        rmd_phs = '|'.join(rmd_node.ph_list[:isa_N])

        if not n_phs or not p_phs or not rmd_phs:
            continue

        order = random.choice([0, 1])

        if order == 0:
            exp_line = f'{n_phs},{p_phs},{rmd_phs}'
            p_id = 0
        else:
            exp_line = f'{n_phs},{rmd_phs},{p_phs}'
            p_id = 1

        if exp_line not in pairs:
            pairs[exp_line] = p_id
            cnt += 1

    if cnt < case_N:
        print(f'[WARN] gen_isa_pairs: chỉ tạo được {cnt}/{case_N} pairs')

    return pairs


# ======================================================
# WRITE INTRUSION CSV
# ======================================================

def write_intrusion(output, intru_maps):
    intru_gold_f = join(output, 'intrusion_gold.txt')

    intru_all = {}

    for tax_name in intru_maps:
        for exp_str in intru_maps[tax_name]:
            intru_all[exp_str] = (
                tax_name,
                intru_maps[tax_name][exp_str]
            )

    each_voter_n = 80
    subset_n = 0
    idx = 0

    intru_exp_f = join(output, f'intrusion_exp_{subset_n}.csv')

    g_exp = open(intru_exp_f, 'w+', encoding='utf-8')
    g_exp.write('0,1,2,3,4,5,outlier id\n')

    with open(intru_gold_f, 'w+', encoding='utf-8') as g_gold:
        for exp_str in intru_all:
            g_exp.write(f'{exp_str},\n')
            g_gold.write(
                '%s\t%s\t%d\n' %
                (
                    exp_str,
                    intru_all[exp_str][0],
                    intru_all[exp_str][1]
                )
            )

            idx += 1

            if idx % each_voter_n == 0:
                g_exp.close()
                subset_n += 1
                intru_exp_f = join(output, f'intrusion_exp_{subset_n}.csv')
                g_exp = open(intru_exp_f, 'w+', encoding='utf-8')
                g_exp.write('0,1,2,3,4,5,outlier id\n')

    g_exp.close()

    print('[SAVE] intrusion_exp_*.csv')
    print('[SAVE]', intru_gold_f)


# ======================================================
# WRITE SUBDOMAIN CSV
# ======================================================

def write_subdomain(output, subdomain_map):
    sub_gold_f = join(output, 'subdomain_gold.txt')

    sub_all = {}

    for tax_name in subdomain_map:
        for exp_str in subdomain_map[tax_name]:
            sub_all[exp_str] = (
                tax_name,
                subdomain_map[tax_name][exp_str]
            )

    each_voter_n = 80
    subset_n = 0
    idx = 0

    sub_exp_f = join(output, f'subdomain_exp_{subset_n}.csv')

    g_exp = open(sub_exp_f, 'w+', encoding='utf-8')
    g_exp.write('child,parent,y or n?\n')

    with open(sub_gold_f, 'w+', encoding='utf-8') as g_gold:
        for exp_str in sub_all:
            g_exp.write(f'{exp_str},\n')
            g_gold.write(
                '%s\t%s\t%d\n' %
                (
                    exp_str,
                    sub_all[exp_str][0],
                    sub_all[exp_str][1]
                )
            )

            idx += 1

            if idx % each_voter_n == 0:
                g_exp.close()
                subset_n += 1
                sub_exp_f = join(output, f'subdomain_exp_{subset_n}.csv')
                g_exp = open(sub_exp_f, 'w+', encoding='utf-8')
                g_exp.write('child,parent,y or n?\n')

    g_exp.close()

    print('[SAVE] subdomain_exp_*.csv')
    print('[SAVE]', sub_gold_f)


# ======================================================
# MAIN HANDLER
# ======================================================

def handler(
    folder,
    output,
    N,
    isa_N,
    intrusion_N,
    subdomain_N,
):
    if not exists(output):
        makedirs(output)

    files = [
        join(folder, f)
        for f in listdir(folder)
        if isfile(join(folder, f)) and f.endswith('.txt')
    ]

    taxs = {}

    for tax_f in files:
        method_name = basename(tax_f)

        print(f'[INFO] Đọc taxonomy: {method_name}')

        taxs[method_name] = read_taxonomy(tax_f)

        n_nodes = len(taxs[method_name].all_nodes)

        print(f'       Số nodes: {n_nodes}')

    if len(taxs) == 0:
        raise ValueError(f'Không tìm thấy taxonomy .txt trong folder: {folder}')

    intru_maps = {}
    subdomain_map = {}

    for tax_name in taxs:
        print(f'\n[GEN] {tax_name}')

        intru_maps[tax_name] = gen_intrusion_pairs(
            taxs[tax_name],
            N,
            intrusion_N
        )

        subdomain_map[tax_name] = gen_subdomain_pairs(
            taxs[tax_name],
            isa_N,
            subdomain_N
        )

        print(f'      intrusion pairs : {len(intru_maps[tax_name])}')
        print(f'      subdomain pairs : {len(subdomain_map[tax_name])}')

    write_intrusion(output, intru_maps)
    write_subdomain(output, subdomain_map)

    print('\n[DONE] Files đã lưu vào:', output)
    print('  intrusion_gold.txt, intrusion_exp_*.csv')
    print('  subdomain_gold.txt, subdomain_exp_*.csv')


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='gen_eval.py',
        description='generate evaluation pairs for taxonomies'
    )

    parser.add_argument(
        '-folder',
        required=True,
        help='Folder chứa generated taxonomies.'
    )

    parser.add_argument(
        '-output',
        required=True,
        help='Folder chứa output files.'
    )

    parser.add_argument(
        '-N',
        type=int,
        default=5,
        help='Số keyword chính trong Topic Intrusion.'
    )

    parser.add_argument(
        '-isa_N',
        type=int,
        default=5,
        help='Số keyword đại diện cho child/parent.'
    )

    parser.add_argument(
        '-intrusion_N',
        type=int,
        default=50,
        help='Số câu Topic Intrusion.'
    )

    parser.add_argument(
        '-subdomain_N',
        type=int,
        default=80,
        help='Số câu Parent-Child.'
    )

    args = parser.parse_args()

    handler(
        folder=args.folder,
        output=args.output,
        N=args.N,
        isa_N=args.isa_N,
        intrusion_N=args.intrusion_N,
        subdomain_N=args.subdomain_N,
    )