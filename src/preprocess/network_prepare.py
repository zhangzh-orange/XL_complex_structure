#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   network_prepare.py
# Time    :   2025/12/12 18:51:38
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================
import pandas as pd
from Bio import SeqIO
import re

# binary_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\WASH_complex\binary_pairs_in_ppi.csv")
# chains_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\chains.csv")
# fasta_file = r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\WASH.fasta"

# network_df = pd.DataFrame(columns=["Chain1","Chain2","Source"])
# mapping = chains_df.set_index("Gene")["Chain"].to_dict()

# network_df["Chain1"] = binary_df["p1"].map(mapping)
# network_df["Chain2"] = binary_df["p2"].map(mapping)

# network_df["Source"] = network_df.apply(
#     lambda row: "".join(sorted([row["Chain1"], row["Chain2"]])),
#     axis=1
# )

# useq_df = pd.DataFrame()
# useq_df["Chain"] = chains_df["Chain"]
# useq_df["Useq"] = chains_df["Gene"]

# gene_seq_map = {}
# for record in SeqIO.parse(fasta_file, "fasta"):
#     desc = record.description
#     seq = str(record.seq)

#     # 使用正则查找 GN=
#     m = re.search(r"GN=([A-Za-z0-9_-]+)", desc)
#     if m:
#         gene = m.group(1)
#         gene_seq_map[gene] = seq

# useq_df["Sequence"] = useq_df["Useq"].map(gene_seq_map)

# print(useq_df)
# network_df.to_csv(r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\network.csv",index=False)
# useq_df.to_csv(r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\useqs.csv",index=False)

def build_network_and_useqs(
    binary_csv,
    chains_csv,
    fasta_file,
    network_out,
    useqs_out
):
    """
    根据 binary PPI、chain 映射和 fasta 文件，
    生成 network.csv 和 useqs.csv

    Parameters
    ----------
    binary_csv : str
        binary_pairs_in_ppi.csv 路径
    chains_csv : str
        chains.csv 路径（包含 Gene 和 Chain）
    fasta_file : str
        fasta 文件路径
    network_out : str
        network.csv 输出路径
    useqs_out : str
        useqs.csv 输出路径
    """

    # 读取数据
    binary_df = pd.read_csv(binary_csv)
    chains_df = pd.read_csv(chains_csv)

    # ========== 构建 network_df ==========
    network_df = pd.DataFrame(columns=["Chain1", "Chain2", "Source"])

    # Gene -> Chain 映射
    mapping = chains_df.set_index("Gene")["Chain"].to_dict()

    network_df["Chain1"] = binary_df["p1"].map(mapping)
    network_df["Chain2"] = binary_df["p2"].map(mapping)

    # Source = 排序后的 Chain 组合
    network_df["Source"] = network_df.apply(
        lambda row: "".join(sorted([row["Chain1"], row["Chain2"]])),
        axis=1
    )

    # ========== 构建 useq_df ==========
    useq_df = pd.DataFrame()
    useq_df["Chain"] = chains_df["Chain"]
    useq_df["Useq"] = chains_df["Gene"]

    # 解析 fasta，建立 Gene -> Sequence 映射
    gene_seq_map = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        desc = record.description
        seq = str(record.seq)

        m = re.search(r"GN=([A-Za-z0-9_-]+)", desc)
        if m:
            gene = m.group(1)
            gene_seq_map[gene] = seq

    useq_df["Sequence"] = useq_df["Useq"].map(gene_seq_map)

    # ========== 输出 ==========
    network_df.to_csv(network_out, index=False)
    useq_df.to_csv(useqs_out, index=False)

    return network_df, useq_df