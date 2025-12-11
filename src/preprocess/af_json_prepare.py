import re
import json
import random
from itertools import combinations
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# ===============================================================
# 1. 工具函数：解析肽段、选择唯一 crosslink
# ===============================================================

def find_bracket_k_position(peptide: str, protein: str):
    """找到肽段 peptide 中 [K] 在 protein 上的定位（1-based）"""
    pep_no_space = peptide.replace(" ", "")
    m = re.search(r"\[([A-Z])\]", pep_no_space)
    if not m:
        return None
    
    aa = m.group(1)
    if aa != "K":
        return None

    aa_index_in_pep = m.start()
    clean_pep = re.sub(r"\[([A-Z])\]", r"\1", pep_no_space)

    start = protein.find(clean_pep)
    if start == -1:
        return None

    return start + aa_index_in_pep + 1  # 1-based


def select_unique_crosslinks(crosslinks):
    """确保每个端点（A,B,C）只能出现一次"""
    shuffled = crosslinks.copy()
    random.shuffle(shuffled)

    chosen = []
    used = set()

    for cl in shuffled:
        p1 = tuple(cl[0])
        p2 = tuple(cl[1])
        if p1 in used or p2 in used:
            continue

        chosen.append(cl)
        used.add(p1)
        used.add(p2)

    return chosen


# ===============================================================
# 2. JSON 构建函数（支持 2mer 和 3mer）
# ===============================================================

def init_json_template(key, proteins):
    """
    proteins: dict like {"A": seqA, "B": seqB, ...}
    """
    seq_block = [{"protein": {"id": tag, "sequence": seq}}
                 for tag, seq in proteins.items()]
    
    return {
        "name": key,
        "modelSeeds": [1],
        "sequences": seq_block,
        "dialect": "alphafold3",
        "version": 1,
        "crosslinks": [{"name": "azide-A-DSBSO", "residue_pairs": []}]
    }


def make_crosslink(tag1, seq1, tag2, seq2, pepA, pepB):
    """生成 residue_pair 结构"""
    return [
        [tag1, find_bracket_k_position(pepA, seq1)],
        [tag2, find_bracket_k_position(pepB, seq2)]
    ]


# ===============================================================
# 3. 交互判断函数（支持 2mer / 3mer）
# ===============================================================

def handle_interaction(
    pA, pB, a_genes, b_genes,
    seq_map, pepA, pepB,
    tagA, tagB,
    crosslinks, seen
):
    """
    pA, pB: protein names
    tagA, tagB: "A","B","C"
    """
    def safe_append(tag1, seq1, tag2, seq2, pepX, pepY):
        cl = make_crosslink(tag1, seq1, tag2, seq2, pepX, pepY)
        if cl[0][1] is None or cl[1][1] is None:
            return

        key = (tuple(cl[0]), tuple(cl[1]))
        if key not in seen:
            seen.add(key)
            crosslinks.append(cl)

    # forward pA → pB
    if pA in a_genes and pB in b_genes:
        safe_append(tagA, seq_map[pA], tagB, seq_map[pB], pepA, pepB)

    # reverse pB → pA
    elif pB in a_genes and pA in b_genes:
        safe_append(tagA, seq_map[pA], tagB, seq_map[pB], pepB, pepA)


# ===============================================================
# 4. 主流程（统一二元组 / 三元组）
# ===============================================================

def prepare_multimer_jsons(
    complex_type,                    # "dimer" or "trimer"
    raw_crosslink_csv,               # raw XL csv
    fasta_file,                      # protein FASTA
    gene_list_excel,                 # Gene → UniProt
    pair_or_triplet_csv,             # "binary_pairs_in_ppi.csv" or "triplet_need_to_pred.csv"
    output_dir,                      # output JSON directory
    sample_times=3
):

    print(f"=== Running: {complex_type} ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------
    # Load data
    # ---------------------------------------
    xl_df = pd.read_csv(raw_crosslink_csv)

    gene_map = pd.read_excel(gene_list_excel).set_index("Gene")["Entry"].to_dict()

    seq_map = {}
    with open(fasta_file) as f:
        for rec in SeqIO.parse(f, "fasta"):
            prot_id = rec.id.split("|")[1]
            seq_map[prot_id] = str(rec.seq)

    # get pairs or triplets
    prot_df = pd.read_csv(pair_or_triplet_csv)
    if complex_type == "dimer":
        complexes = [tuple(row) for row in prot_df[["p1", "p2"]].values]
    else:
        complexes = [tuple(row) for row in prot_df[["p1", "p2", "p3"]].values]

    # ---------------------------------------
    # Iterate
    # ---------------------------------------
    json_files = {}

    for sample_time in range(sample_times):
        for comp in complexes:

            # ====== 构造 key ======
            key = "_".join(list(comp) + [str(sample_time)])

            # ====== 生成 tag → protein map ======
            tags = ["A", "B", "C"]
            prot_map = {}
            for t, p in zip(tags, comp):
                prot_map[t] = seq_map[gene_map[p]]

            # ====== JSON 初始化 ======
            json_files[key] = init_json_template(key, prot_map)
            crosslinks = json_files[key]["crosslinks"][0]["residue_pairs"]

            # ====== crosslink 搜索 ======
            seen = set()

            for _, row in xl_df.iterrows():
                # 跳过空值
                if any(pd.isna(v) or str(v).strip() == "" for v in [row.gene_a, row.gene_b, row.pepA, row.pepB]):
                    continue

                a_genes = {g.strip() for g in row.gene_a.split(";")}
                b_genes = {g.strip() for g in row.gene_b.split(";")}

                pepA = row.pepA
                pepB = row.pepB

                # ========== dimer ==========
                if complex_type == "dimer":
                    p1, p2 = comp
                    handle_interaction(
                        p1, p2, a_genes, b_genes,
                        seq_map={p: seq_map[gene_map[p]] for p in comp},
                        pepA=pepA, pepB=pepB,
                        tagA="A", tagB="B",
                        crosslinks=crosslinks, seen=seen
                    )

                # ========== trimer ==========
                else:
                    p1, p2, p3 = comp
                    seq_local = {p: seq_map[gene_map[p]] for p in comp}

                    # p1-p2
                    handle_interaction(p1, p2, a_genes, b_genes, seq_local, pepA, pepB, "A", "B",
                                       crosslinks, seen)
                    # p1-p3
                    handle_interaction(p1, p3, a_genes, b_genes, seq_local, pepA, pepB, "A", "C",
                                       crosslinks, seen)
                    # p2-p3
                    handle_interaction(p2, p3, a_genes, b_genes, seq_local, pepA, pepB, "B", "C",
                                       crosslinks, seen)

            # 过滤唯一 crosslinks
            json_files[key]["crosslinks"][0]["residue_pairs"] = \
                select_unique_crosslinks(crosslinks)

    # ---------------------------------------
    # Save JSON
    # ---------------------------------------
    for key, data in json_files.items():
        out = Path(output_dir) / f"{key}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✔ Completed: {len(json_files)} JSON files saved to {output_dir}")


# ===============================================================
# 5. 调用示例（LRBA/SNARE 与 CORVET 都可跑）
# ===============================================================

if __name__ == "__main__":

    # ====== dimer ======
    # prepare_multimer_jsons(
    #     complex_type="dimer",
    #     raw_crosslink_csv=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\heklopit_pl3017_frd1ppi_sc151_fdr1rp_LRBAandSNARE.csv",
    #     fasta_file=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\LRBAandSNARE_small.fasta",
    #     gene_list_excel=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\LRBAandSNARE_gene_list.xlsx",
    #     pair_or_triplet_csv=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\binary_pairs_in_ppi.csv",
    #     output_dir=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\jsons_2mer",
    # )

    # # ====== trimer ======
    prepare_multimer_jsons(
        complex_type="trimer",
        raw_crosslink_csv=r"N:\08_NK_structure_prediction\data\Exocyst_complex\heklopit_pl3017_frd1ppi_sc151_fdr1rp_Exocyst.csv",
        fasta_file=r"N:\08_NK_structure_prediction\data\Exocyst_complex\Exocyst.fasta",
        gene_list_excel=r"N:\08_NK_structure_prediction\data\Exocyst_complex\Exocyst_gene_list.xlsx",
        pair_or_triplet_csv=r"N:\08_NK_structure_prediction\data\Exocyst_complex\triplet_need_to_pred.csv",
        output_dir=r"N:\08_NK_structure_prediction\data\Exocyst_complex\jsons_3mer",
    )
    pass
