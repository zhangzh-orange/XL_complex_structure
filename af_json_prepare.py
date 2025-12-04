
def find_bracket_k_position(peptide, protein):
    import re

    # 去除 peptide 中所有空格
    pep_no_space = peptide.replace(" ", "")

    # 匹配 [K]，其他情况直接 None
    m = re.search(r"\[([A-Z])\]", pep_no_space)
    if not m:
        return None  # 没有任何 [X]
    
    aa = m.group(1)
    if aa != "K":
        # print(peptide)
        return None  # 不是 [K]，直接返回 None

    aa_index_in_pep = m.start()     # '[' 在序列中的位置 (0-based)

    # 去除括号
    clean_pep = re.sub(r"\[([A-Z])\]", r"\1", pep_no_space)

    # 查找肽段位置
    start = protein.find(clean_pep)
    if start == -1:
        return None  # 肽段无法定位到蛋白

    # 计算蛋白上的氨基酸位置 (0-based → 1-based)
    aa_pos_1 = start + aa_index_in_pep + 1

    return aa_pos_1

import random

def select_unique_crosslinks(crosslinks):
    """
    crosslinks: 双层列表，例如 [[["A", 291], ["B", 336]], ...]
    
    返回：
        - 保证每个端点只出现一次
        - 随机性，每次运行结果可能不同
    """
    crosslinks_shuffled = crosslinks.copy()
    random.shuffle(crosslinks_shuffled)  # 打乱顺序增加随机性

    chosen = []
    used_endpoints = set()

    for cl in crosslinks_shuffled:
        e1 = tuple(cl[0])
        e2 = tuple(cl[1])

        # 如果任意一个端点已被占用，则跳过
        if e1 in used_endpoints or e2 in used_endpoints:
            continue

        # 两端都未使用 → append 并标记端点
        chosen.append(cl)
        used_endpoints.add(e1)
        used_endpoints.add(e2)

    return chosen



import pandas as pd

copi_complex_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\CORVET_complex\heklopit_pl3017_frd1ppi_sc151_fdr1rp_CORVET.csv")
# print(copi_complex_df.head())


from Bio import SeqIO

copi_fasta_path = r"N:\08_NK_structure_prediction\data\CORVET_complex\CORVET.fasta"

copi_seqs_dict = {}
with open(copi_fasta_path) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        protein_id = str(record.id).split("|")[1]
        seq = str(record.seq)
        copi_seqs_dict[protein_id] = seq

# print(copi_seqs_dict)


protein_gene_df = pd.read_excel(r"N:\08_NK_structure_prediction\data\CORVET_complex\CORVET_gene_list.xlsx",sheet_name=0)
gene_protein_map = protein_gene_df.set_index("Gene")["Entry"].to_dict()
# print(gene_protein_map)

# 读取蛋白三元组
prot_triplet_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\CORVET_complex\triplet_need_to_pred.csv")
triplet_list = [tuple(row) for row in prot_triplet_df[["p1", "p2", "p3"]].values]
# print(triplet_list)

json_files = {}

def make_crosslink(pair_label1, p1_seq, pair_label2, p2_seq, pepA, pepB):
    """生成一个 residue_pair 结构"""
    return [ [pair_label1, find_bracket_k_position(pepA, p1_seq)],
              [pair_label2, find_bracket_k_position(pepB, p2_seq)] ]


for sample_time in range(3):
    for (p1, p2, p3) in triplet_list:
        key = f"{p1}_{p2}_{p3}_{sample_time}"
        # print(key)
        json_files[key] = {
            "name": key,
            "modelSeeds": [1],
            "sequences": [
                {"protein":{"id": "A", "sequence": copi_seqs_dict[gene_protein_map[p1]]}},
                {"protein":{"id": "B", "sequence": copi_seqs_dict[gene_protein_map[p2]]}},
                {"protein":{"id": "C", "sequence": copi_seqs_dict[gene_protein_map[p3]]}},
            ],
            "dialect": "alphafold3",
            "version": 1,
            "crosslinks": [{"name": "azide-A-DSBSO", "residue_pairs": []}]
        }
        
        crosslinks = json_files[key]["crosslinks"][0]["residue_pairs"]

        seen_keys = set()
        
        for index, row in copi_complex_df.iterrows():
            a_gene, b_gene = row["gene_a"], row["gene_b"]
            pepA, pepB = row["pepA"], row["pepB"]

            # 任意字段为空 → 跳过
            if any(pd.isna(v) or str(v).strip()=="" for v in [a_gene, b_gene, pepA, pepB]):
                continue

            a_genes = {g.strip() for g in a_gene.split(";")}
            b_genes = {g.strip() for g in b_gene.split(";")}

            def safe_append(*args):
                cl = make_crosslink(*args)
                if not cl:
                    return

                # cl example: [["A", 11], ["B", 46]]
                # If either side contains None → do NOT append
                if cl[0][1] is None or cl[1][1] is None:
                    return
                
                key = (tuple(cl[0]), tuple(cl[1]))

                if key not in seen_keys:
                    seen_keys.add(key)
                    crosslinks.append(cl)

            # p1 - p2
            if p1 in a_genes and p2 in b_genes:
                safe_append("A", copi_seqs_dict[gene_protein_map[p1]],
                                "B", copi_seqs_dict[gene_protein_map[p2]],
                                pepA, pepB)

            elif p2 in a_genes and p1 in b_genes:
                safe_append(
                                "A", copi_seqs_dict[gene_protein_map[p1]],
                                "B", copi_seqs_dict[gene_protein_map[p2]], pepB,
                                pepA)

            # p1 - p3
            elif p1 in a_genes and p3 in b_genes:
                safe_append("A", copi_seqs_dict[gene_protein_map[p1]],
                                "C", copi_seqs_dict[gene_protein_map[p3]],
                                pepA, pepB)

            elif p3 in a_genes and p1 in b_genes:
                safe_append(
                                "A", copi_seqs_dict[gene_protein_map[p1]],
                                "C", copi_seqs_dict[gene_protein_map[p3]], pepB,
                                pepA)

            # p2 - p3
            elif p2 in a_genes and p3 in b_genes:
                safe_append("B", copi_seqs_dict[gene_protein_map[p2]],
                                "C", copi_seqs_dict[gene_protein_map[p3]],
                                pepA, pepB)

            elif p3 in a_genes and p2 in b_genes:
                safe_append(
                                "B", copi_seqs_dict[gene_protein_map[p2]],
                                "C", copi_seqs_dict[gene_protein_map[p3]], pepB,
                                pepA)
                
        crosslinks[:] = select_unique_crosslinks(crosslinks)
        # break


    # 输出
    # print(json_files)
    import json
    for key in json_files.keys():
        file_name = rf"N:\08_NK_structure_prediction\data\CORVET_complex\jsons\{key}.json"
        data = json_files[key]
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
