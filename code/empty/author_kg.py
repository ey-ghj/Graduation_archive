#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 19:13
# @Author  : Cc
# @Site    : 
# @Version : 
# @File    : author_kg.py
# @Software: PyCharm
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'

import pandas as pd
# 读取第一个CSV文件
df1 = pd.read_csv('../source1.csv', encoding='gbk')
# 读取第二个CSV文件
df2 = pd.read_csv('../source2.csv', encoding='gbk')
# 合并两个DataFrame对象
merged_df = pd.concat([df1, df2])

# 删除包含NaN值的行
merged_df = merged_df.dropna(subset=['Author-作者'])
# 提取高频作者列表
authors = merged_df['Author-作者'].str.replace(';', ',').str.split(',')
author_count = {}
for alist in authors:
    # 检查每个作者是否为空
    non_empty_authors = [author.strip().strip('"') for author in alist if author.strip().strip('"') != '']
    # 对每个非空作者计数
    for author in non_empty_authors:
        if author in author_count:
            author_count[author] += 1
        else:
            author_count[author] = 1

# 获取所有出现次数大于1的作者
top_authors = sorted([author for author, count in author_count.items() if count > 1], key=author_count.get, reverse=True)

# 将高频作者列表写入CSV文件
result_df = pd.DataFrame({'Value': top_authors,
                          'Frequency': [author_count[a] for a in top_authors]})
result_df['Percentage'] = result_df['Frequency']/len(merged_df)
result_df.to_csv('./output/top_authors.csv', index=False, encoding='utf-8-sig', header=['Value', 'Frequency', 'Percentage'])

# 基于作者之间的共同出现次数构建知识图谱
G = nx.Graph()
for alist in authors:
    # 检查每个作者是否为空
    non_empty_authors = [author.strip().strip('"') for author in alist if author.strip().strip('"') != '']
    # 对每对非空作者增加一个权重
    for i in range(len(non_empty_authors)):
        for j in range(i+1, len(non_empty_authors)):
            if G.has_edge(non_empty_authors[i], non_empty_authors[j]):
                # 如果边已存在，则权重加1
                G[non_empty_authors[i]][non_empty_authors[j]]['weight'] += 1
            else:
                # 否则，添加一条新的边，并将权重设置为1
                G.add_edge(non_empty_authors[i], non_empty_authors[j], weight=1)

# 绘制知识图谱
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8, font_family='Microsoft YaHei')
plt.axis('off')
plt.savefig('./output/author_knowledge_graph.png')
plt.show()