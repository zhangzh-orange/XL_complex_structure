import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import os


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

def clean_residue_pair_file(input_path, manual_map):
    df = pd.read_csv(input_path)

    for col in ["gene_a", "gene_b"]:
        if col not in df.columns:
            raise ValueError(f"列 {col} 不存在于文件中")

        # 1️⃣ 统一格式：分号两边只保留一个空格
        df[col] = (
            df[col]
            .astype(str)
            .str.split(";")
            .apply(lambda x: "; ".join(s.strip() for s in x))
        )

        # 2️⃣ 使用 replace（不是 map）
        df[col] = df[col].replace(manual_map)

    # ====== 输出 ======
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(os.path.dirname(input_path),
                            base_name + "_cleaned.csv")

    df.to_csv(out_path, index=False)
    print("清洗完成：", out_path)

# -------------------------------------------------------------
# 6. 主流程
# -------------------------------------------------------------
def main():

    # ====== 文件路径 ======
    input_path = r"N:\08_NK_structure_prediction\data\WASH_complex\heklopit_pl3017_frd1ppi_sc151_fdr1rp_WASH.csv"
    out_folder = r"N:\08_NK_structure_prediction\data\WASH_complex"

    # ====== 读取数据并构建网络 ======
    df = load_interaction_data(input_path)
    G = build_ppi_network(df)

    # ====== 打印需要清洗的节点 ======
    dirty_nodes = [n for n in G.nodes() if ";" in str(n)]
    print("Nodes needing cleanup:", dirty_nodes)

    # ====== 清洗节点名称 ======
    manual_map = {
        'WASH3P; WASH3P; WASH2P; WASH2P; WASHC1; WASHC1':"WASHC1", 
        'WASHC2A; WASHC2C; WASHC2A; WASHC2C; WASHC2A; WASHC2C':"WASHC2C", 
        'WASH3P; WASH2P; WASHC1':"WASHC1", 
        'WASHC4; WASHC4; WASHC4':"WASHC4", 
        'WASHC5; WASHC5; WASHC5':"WASHC5", 
        'WASHC3; WASHC3; WASHC3':"WASHC3", 
        'FKBP15; FKBP15; FKBP15':"FKBP15", 
        'WASHC2A; WASHC2A; WASHC2A; WASHC2C; WASHC2C; WASHC2C':"WASHC2C", 
        'WASH3P; WASH2P; WASHC1; WASH3P; WASH2P; WASHC1':"WASHC1", 
        'WASHC2A; WASHC2C':"WASHC2C", 
        'WASHC4; WASHC4':"WASHC4", 
        'WASH2P; WASHC1':"WASHC1", 
        'WASHC2A; WASHC2A; WASHC2C; WASHC2C':"WASHC2C", 
        'WASH2P; WASHC1; WASH2P; WASHC1':"WASHC1", 
        'WASHC3; WASHC3':"WASHC3"
    }
    G = clean_node_names(G, manual_map)

    # ====== 绘图 ======
    plot_ppi_network(G, "COPI Complex PPI Network (Cleaned Names)")

    # 重新清理原文件
    clean_residue_pair_file(input_path, manual_map)

    # ====== 复合体分析 ======
    protein_list = ["WASHC1",
                    "WASHC2C",
                    "WASHC3",
                    "WASHC4",
                    "WASHC5",
                    "ENTR1",
                    "FKBP15"
                    ]
    binary_pairs_in_ppi, triplet_in_ppi = analyze_complexes(G, protein_list)

    # ====== 输出结果 ======
    df_binary_ppi = pd.DataFrame(set(binary_pairs_in_ppi), columns=["p1", "p2"])
    df_binary_ppi.to_csv(os.path.join(out_folder,"binary_pairs_in_ppi.csv"), index=False)

    df_triplet_ppi = pd.DataFrame(set(triplet_in_ppi), columns=["p1", "p2", "p3"])
    df_triplet_ppi.to_csv(os.path.join(out_folder,"triplet_need_to_pred.csv"), index=False)

    print(f"Total triplets: {len(list(combinations(protein_list, 3)))}")
    print(f"Triplets found in PPI: {len(triplet_in_ppi)}")


if __name__ == "__main__":
    main()
