import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

xl_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\COPI_complex\heklopit_pl3017_frd1ppi_sc151_fdr1rp_COPI.csv")
xl_df.head()


G = nx.Graph()
for index, row in xl_df.iterrows():
    # 分割蛋白质列表（如果有多个蛋白质用分号分隔）
    proteins_a = [p.strip() for p in str(row["gene_a"]).split(";")]
    proteins_b = [p.strip() for p in str(row["gene_b"]).split(";")]
    
    # 检查是否有交集（即是否有相同的蛋白质）
    # 如果有交集，说明是自环或内部连接，跳过
    if set(proteins_a) & set(proteins_b):
        continue  # 跳过这一行，不添加边
    
    # 如果没有交集，添加边
    G.add_edge(row["gene_a"], row["gene_b"])

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, 
                 pos=pos,
                 with_labels=True,
                 node_size=500,
                 node_color='lightblue',
                 edge_color='gray',
                 font_size=10,
                 font_weight='bold',
                 width=1.0,
                 alpha=0.8)

plt.title("COPI Complex PPI Network (Undirected)", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# 显示图中仍带有;符号的节点名，修改节点名
print([prot for prot in G.nodes() if ";" in prot])

single_prot_map = {
    'ASS1; ARCN1':'ARCN1', 
    'COPG2; COPG1':'COPG1',  # g1z1
    'ARF4; ARF6; ARF1; ARF5':'ARF1'
}

# 创建重命名映射
rename_dict = {}
for node in G.nodes():
    # 如果在映射字典中，就替换
    if node in single_prot_map:
        rename_dict[node] = single_prot_map[node]
    # 否则保留原样
    else:
        # 可以添加更多清理逻辑，比如如果有分号，取第一个
        if ";" in str(node):
            rename_dict[node] = str(node).split(";")[0].strip()

# 重命名节点
G = nx.relabel_nodes(G, rename_dict)

# 绘制图形
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, 
                 pos=pos,
                 with_labels=True,
                 node_size=500,
                 node_color='lightblue',
                 edge_color='gray',
                 font_size=10,
                 font_weight='bold',
                 width=1.0,
                 alpha=0.8)

plt.title("COPI Complex PPI Network (Cleaned Names)", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()


from itertools import combinations

protein_list = ['COPB1',
 'COPB2',
 'ARCN1',
 'ARFGAP2',
 'COPA',
#  'COPG2', # g1z1
 'COPZ1',
 'COPG1',
 'COPE',
 'ARFGAP3',
 'CCDC115',
 'TMEM199',
 'ARF1']

# 将所有有连接关系的三元组进行结构预测
triple_complexes = list(combinations(protein_list, 3))
# triple_complexes

# 检测三元组在相互作用组中的连接（拆分为二元组）
ppi_set = {frozenset(ppi) for ppi in G.edges()}

triplet_need_to_pred = []

for triplet in triple_complexes:
    p1, p2, p3 = triplet
    
    # 三个成对组合
    pairs = [
        frozenset((p1, p2)),
        frozenset((p1, p3)),
        frozenset((p2, p3)),
    ]
    
    # 统计三元组中有几对在 PPI 中
    link_count = sum(pair in ppi_set for pair in pairs)
    
    if link_count >= 2:
        triplet_need_to_pred.append(triplet)

print(len(triple_complexes))
print(len(triplet_need_to_pred))

import pandas as pd

df_triplet = pd.DataFrame(triplet_need_to_pred, columns=["p1", "p2", "p3"])
df_triplet.to_csv(r"N:\08_NK_structure_prediction\data\COPI_complex\triplet_need_to_pred.csv", index=False)