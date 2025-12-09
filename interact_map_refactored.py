import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


# -------------------------------------------------------------
# 1. 数据读取
# -------------------------------------------------------------
def load_interaction_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# -------------------------------------------------------------
# 2. 构建 PPI 网络
# -------------------------------------------------------------
def build_ppi_network(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in df.iterrows():
        proteins_a = [p.strip() for p in str(row["gene_a"]).split(";")]
        proteins_b = [p.strip() for p in str(row["gene_b"]).split(";")]

        # 跳过自环或内部连接
        if set(proteins_a) & set(proteins_b):
            continue

        G.add_edge(row["gene_a"], row["gene_b"])

    return G


# -------------------------------------------------------------
# 3. 清洗节点名称
# -------------------------------------------------------------
def clean_node_names(G: nx.Graph, manual_map: dict | None = None) -> nx.Graph:
    manual_map = manual_map or {}
    rename_dict = {}

    for node in G.nodes():
        if node in manual_map:
            rename_dict[node] = manual_map[node]
        elif ";" in str(node):
            rename_dict[node] = node.split(";")[0].strip()

    return nx.relabel_nodes(G, rename_dict)


# -------------------------------------------------------------
# 4. 绘图函数（可复用）
# -------------------------------------------------------------
def plot_ppi_network(G: nx.Graph, title: str, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_size=500,
        node_color='lightblue',
        edge_color='gray',
        font_size=10,
        font_weight='bold',
        width=1.0,
        alpha=0.8
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# 5. 检测三元组复合体与二元交互
# -------------------------------------------------------------
def analyze_complexes(G: nx.Graph, protein_list: list[str]):
    binary_pairs = list(combinations(protein_list, 2))
    triple_complexes = list(combinations(protein_list, 3))

    ppi_set = {frozenset(edge) for edge in G.edges()}

    binary_pairs_in_ppi = []
    triplet_in_ppi = []

    for triplet in triple_complexes:
        p1, p2, p3 = triplet
        pairs = [
            frozenset((p1, p2)),
            frozenset((p1, p3)),
            frozenset((p2, p3)),
        ]

        # 二元组存在于 PPI
        for pair in pairs:
            if pair in ppi_set:
                binary_pairs_in_ppi.append(tuple(pair))

        # 三元组内至少 2 对存在于 PPI
        if sum(pair in ppi_set for pair in pairs) >= 2:
            triplet_in_ppi.append(triplet)

    return binary_pairs_in_ppi, triplet_in_ppi


# -------------------------------------------------------------
# 6. 主流程
# -------------------------------------------------------------
def main():

    # ====== 文件路径 ======
    input_path = r"N:\08_NK_structure_prediction\data\LRBAandSNARE\heklopit_pl3017_frd1ppi_sc151_fdr1rp_LRBAandSNARE.csv"
    out_binary_path = r"N:\08_NK_structure_prediction\data\LRBAandSNARE\binary_pairs_in_ppi.csv"

    # ====== 读取数据并构建网络 ======
    df = load_interaction_data(input_path)
    G = build_ppi_network(df)

    # ====== 打印需要清洗的节点 ======
    dirty_nodes = [n for n in G.nodes() if ";" in str(n)]
    print("Nodes needing cleanup:", dirty_nodes)

    # ====== 清洗节点名称 ======
    # manual_map = {
    #     'ASS1; ARCN1': 'ARCN1',
    #     'COPG2; COPG1': 'COPG1',
    #     'ARF4; ARF6; ARF1; ARF5': 'ARF1'
    # }
    # G = clean_node_names(G, manual_map)

    # ====== 绘图 ======
    plot_ppi_network(G, "COPI Complex PPI Network (Cleaned Names)")

    # ====== 复合体分析 ======
    protein_list = ["LRBA", "STX7", "STX12"]
    binary_pairs_in_ppi, triplet_in_ppi = analyze_complexes(G, protein_list)

    # ====== 输出结果 ======
    df_binary_ppi = pd.DataFrame(set(binary_pairs_in_ppi), columns=["p1", "p2"])
    df_binary_ppi.to_csv(out_binary_path, index=False)

    print(f"Total triplets: {len(list(combinations(protein_list, 3)))}")
    print(f"Triplets found in PPI: {len(triplet_in_ppi)}")


if __name__ == "__main__":
    main()
