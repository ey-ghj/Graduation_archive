import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取第一个CSV文件
df1 = pd.read_csv('../source1.csv', encoding='gbk')

# 读取第二个CSV文件
df2 = pd.read_csv('../source2.csv', encoding='gbk')


# 合并数据
merged_df = pd.concat([df1, df2])

# 筛选机构
organ_count = merged_df['Organ-单位'].str.split(';').explode().value_counts()
organ_count_filtered = organ_count[organ_count <= 200]
org_list = list(organ_count_filtered.index)

# 创建空白有向图
G = nx.DiGraph()

# 添加节点
for org in org_list:
    G.add_node(org)
# 添加边
for index, row in merged_df.iterrows():
    orgs = row['Organ-单位']
    if pd.isna(orgs):
        continue
    orgs = orgs.split(';')
    for i in range(len(orgs)):
        if orgs[i] not in org_list:
            continue
        for j in range(i+1, len(orgs)):
            if orgs[j] not in org_list:
                continue
            if G.has_edge(orgs[i], orgs[j]):
                G.edges[orgs[i], orgs[j]]['weight'] += 1
            else:
                G.add_edge(orgs[i], orgs[j], weight=1)

# 绘制知识图谱
pos = nx.spring_layout(G, k=0.15, iterations=20)
plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(G, pos, node_size=1000, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=[d['weight']*0.1 for (u, v, d) in G.edges(data=True)])
nx.draw_networkx_labels(G, pos, font_size=14, font_family='SimHei')
plt.axis('off')
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
plt.show()
