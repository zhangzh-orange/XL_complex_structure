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

binary_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\CORVET_complex\binary_pairs_in_ppi.csv")
chains_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\chains.csv")
fasta_file = r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\CORVET.fasta"

network_df = pd.DataFrame(columns=["Chain1","Chain2","Source"])
mapping = chains_df.set_index("Gene")["Chain"].to_dict()

network_df["Chain1"] = binary_df["p1"].map(mapping)
network_df["Chain2"] = binary_df["p2"].map(mapping)

network_df["Source"] = network_df.apply(
    lambda row: "".join(sorted([row["Chain1"], row["Chain2"]])),
    axis=1
)

useq_df = pd.DataFrame()
useq_df["Chain"] = chains_df["Chain"]
useq_df["Useq"] = chains_df["Gene"]

gene_seq_map = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    desc = record.description
    seq = str(record.seq)

    # 使用正则查找 GN=
    m = re.search(r"GN=([A-Za-z0-9_-]+)", desc)
    if m:
        gene = m.group(1)
        gene_seq_map[gene] = seq

useq_df["Sequence"] = useq_df["Useq"].map(gene_seq_map)

print(useq_df)
network_df.to_csv(r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\network.csv",index=False)
useq_df.to_csv(r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\useqs.csv",index=False)