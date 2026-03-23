import os
from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------------------------------------------
# 1. Data loading
# -------------------------------------------------------------
def load_interaction_data(path: str) -> pd.DataFrame:
    """
    Load an XL-MS interaction table from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.  The file must contain at least the columns
        ``gene_a`` and ``gene_b`` (used by ``build_ppi_network``).

    Returns
    -------
    pd.DataFrame
        Raw interaction table with all original columns preserved.
    """
    return pd.read_csv(path)


# -------------------------------------------------------------
# 2. Build PPI network
# -------------------------------------------------------------
def build_ppi_network(df: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected PPI graph from a crosslink / interaction table.

    Rows where ``gene_a`` and ``gene_b`` share any protein name (self-loops)
    are skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Interaction table with columns ``gene_a`` and ``gene_b``.

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        proteins_a = {p.strip() for p in str(row["gene_a"]).split(";")}
        proteins_b = {p.strip() for p in str(row["gene_b"]).split(";")}
        if proteins_a & proteins_b:
            continue
        G.add_edge(row["gene_a"], row["gene_b"])
    return G


# -------------------------------------------------------------
# 3. Clean node names
# -------------------------------------------------------------
def clean_node_names(G: nx.Graph, manual_map: dict | None = None) -> nx.Graph:
    """
    Rename graph nodes using a manual mapping and/or by taking the first
    token before a semicolon.

    Parameters
    ----------
    G : nx.Graph
    manual_map : dict, optional
        ``{old_name: new_name}`` overrides applied before the semicolon rule.

    Returns
    -------
    nx.Graph
    """
    manual_map = manual_map or {}
    rename_dict = {}
    for node in G.nodes():
        if node in manual_map:
            rename_dict[node] = manual_map[node]
        elif ";" in str(node):
            rename_dict[node] = node.split(";")[0].strip()
    return nx.relabel_nodes(G, rename_dict)


# -------------------------------------------------------------
# 4. Visualization
# -------------------------------------------------------------
def plot_ppi_network(G: nx.Graph, title: str, figsize=(12, 10)):
    """Draw the PPI network with spring layout."""
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        edge_color="gray",
        font_size=10,
        font_weight="bold",
        width=1.0,
        alpha=0.8,
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# 5. Enumerate dimers and trimers
# -------------------------------------------------------------
def analyze_complexes(G: nx.Graph, protein_list: list[str]):
    """
    Enumerate binary pairs and trimers from ``protein_list`` that are
    supported by the PPI network.

    A trimer is included when at least two of its three pairwise edges
    exist in the network.

    Parameters
    ----------
    G : nx.Graph
    protein_list : list[str]
        Proteins of interest.

    Returns
    -------
    tuple[list, list]
        ``(binary_pairs_in_ppi, triplet_in_ppi)``
    """
    ppi_set = {frozenset(edge) for edge in G.edges()}

    binary_pairs_in_ppi = []
    triplet_in_ppi = []

    for triplet in combinations(protein_list, 3):
        p1, p2, p3 = triplet
        pairs = [
            frozenset((p1, p2)),
            frozenset((p1, p3)),
            frozenset((p2, p3)),
        ]
        for pair in pairs:
            if pair in ppi_set:
                binary_pairs_in_ppi.append(tuple(pair))
        if sum(pair in ppi_set for pair in pairs) >= 2:
            triplet_in_ppi.append(triplet)

    return binary_pairs_in_ppi, triplet_in_ppi


# -------------------------------------------------------------
# 6. Clean residue-pair CSV
# -------------------------------------------------------------
def clean_residue_pair_file(input_path: str, manual_map: dict):
    """
    Normalise ``gene_a`` / ``gene_b`` columns in a raw XL-MS CSV using
    ``manual_map``, then write the result to ``<original>_cleaned.csv``.

    Parameters
    ----------
    input_path : str
        Path to the raw residue-pair CSV.
    manual_map : dict
        Mapping of raw (possibly merged) names to clean gene names.
    """
    df = pd.read_csv(input_path)

    for col in ["gene_a", "gene_b"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_path}")
        df[col] = (
            df[col]
            .astype(str)
            .str.split(";")
            .apply(lambda x: "; ".join(s.strip() for s in x))
        )
        df[col] = df[col].replace(manual_map)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(os.path.dirname(input_path), base_name + "_cleaned.csv")
    df.to_csv(out_path, index=False)
    print("Cleaned file written to:", out_path)
