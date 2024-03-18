import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取第一个CSV文件
df1 = pd.read_csv('source1.csv', encoding='gbk')

# 读取第二个CSV文件
df2 = pd.read_csv('source2.csv', encoding='gbk')

# 合并两个DataFrame对象
merged_df = pd.concat([df1, df2])

# 统计机构出现频次并筛选
organ_count = merged_df['Organ-单位'].str.split(';').explode().value_counts()
organ_count_filtered = organ_count[organ_count <= 200]
org_list = list(organ_count_filtered.index)

# 创建无向图对象
G = nx.Graph()

# 遍历文献数据集，添加节点和计算节点权重
node_weight_dict = dict()
for _, row in merged_df.iterrows():
    organs_str = str(row['Organ-单位'])  # 将机构信息转换为字符串
    if organs_str.strip() == '':  # 如果机构信息为空，则跳过本行数据
        continue
    organs = organs_str.split(';')
    added_orgs = set()
    for i, org in enumerate(organs):
        if org in org_list:
            if org not in node_weight_dict:
                node_weight_dict[org] = 0
            node_weight_dict[org] += 1 / (i + 1)  # 计算节点权重
            added_orgs.add(org)
    for org1 in added_orgs:
        for org2 in added_orgs:
            if org1 != org2:
                if G.has_edge(org1, org2):
                    G[org1][org2]['weight'] += 1
                else:
                    G.add_edge(org1, org2, weight=1)

# 筛选边
edge_list = []
for u, v, d in G.edges(data=True):
    if d['weight'] >= 5:  # 只保留权重大于等于5的边
        edge_list.append((u, v, d['weight']))



# 获取每个节点的度数并排序
degrees = dict(G.degree())
sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

# 仅保留前50个节点
top_nodes = [node for node, degree in sorted_degrees[:50]]

# 创建子图，并绘制知识图谱
subgraph = G.subgraph(top_nodes)
pos = nx.spring_layout(subgraph, k=0.15, iterations=20)
plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(subgraph, pos, node_size=1000, alpha=0.8)
nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=[d['weight']*0.1 for (u, v, d) in subgraph.edges(data=True)])
nx.draw_networkx_labels(subgraph, pos, font_size=14, font_family='SimHei')
plt.axis('off')
plt.savefig("./output/organ_kg3.png", dpi=300)
nx.write_gexf(G, "./output/organ_graph3.gexf")
plt.show()


# 获取孤立节点列表
isolates = list(nx.isolates(G))

# 删除孤立节点
G.remove_nodes_from(isolates)

# 绘制知识图谱（与原代码一致）
pos = nx.spring_layout(G, k=0.15, iterations=20)
plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(G, pos, node_size=1000, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=[d['weight']*0.1 for (u, v, d) in G.edges(data=True)])
nx.draw_networkx_labels(G, pos, font_size=14, font_family='SimHei')
plt.axis('off')
plt.savefig("./output/organ_kg2.png", dpi=300)
nx.write_gexf(G, "./output/organ_graph2.gexf")
plt.show()
