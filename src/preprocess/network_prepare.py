#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   network_prepare.py
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================
import re

import pandas as pd
from Bio import SeqIO


def build_network_and_useqs(
    binary_csv,
    chains_csv,
    fasta_file,
    network_out,
    useqs_out,
):
    """
    Build ``network.csv`` and ``useqs.csv`` from binary PPI pairs, chain
    mapping, and a protein FASTA file.

    Parameters
    ----------
    binary_csv : str
        Path to CSV with columns ``p1`` and ``p2`` (binary protein pairs).
    chains_csv : str
        Path to ``chains.csv`` with columns ``Gene`` and ``Chain``.
    fasta_file : str
        Path to FASTA file whose headers contain ``GN=<gene_name>`` fields.
    network_out : str
        Output path for ``network.csv``.
    useqs_out : str
        Output path for ``useqs.csv``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(network_df, useq_df)``
    """
    binary_df = pd.read_csv(binary_csv)
    chains_df = pd.read_csv(chains_csv)

    # --- network_df ---
    mapping = chains_df.set_index("Gene")["Chain"].to_dict()
    network_df = pd.DataFrame(columns=["Chain1", "Chain2", "Source"])
    network_df["Chain1"] = binary_df["p1"].map(mapping)
    network_df["Chain2"] = binary_df["p2"].map(mapping)
    network_df["Source"] = network_df.apply(
        lambda row: "".join(sorted([row["Chain1"], row["Chain2"]])),
        axis=1,
    )

    # --- useq_df ---
    useq_df = pd.DataFrame()
    useq_df["Chain"] = chains_df["Chain"]
    useq_df["Useq"] = chains_df["Gene"]

    gene_seq_map = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        m = re.search(r"GN=([A-Za-z0-9_-]+)", record.description)
        if m:
            gene_seq_map[m.group(1)] = str(record.seq)

    useq_df["Sequence"] = useq_df["Useq"].map(gene_seq_map)

    network_df.to_csv(network_out, index=False)
    useq_df.to_csv(useqs_out, index=False)

    return network_df, useq_df
