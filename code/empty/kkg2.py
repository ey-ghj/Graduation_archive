#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/5 15:28
# @Author  : Cc
# @Site    : 
# @Version : 
# @File    : kkg2.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.sparse import coo_matrix
# 读取两个CSV文件，合并成一个DataFrame对象
df1 = pd.read_csv('../source1.csv', encoding='gbk')
df2 = pd.read_csv('../source2.csv', encoding='gbk')
merged_df = pd.concat([df1, df2])

# 抽取'Keyword-关键词'字段信息，并以';'进行分割
keywords_series = merged_df['Keyword-关键词'].str.split(';')

# 拆分之后的每一个元素都是一个列表，使用explode()方法将其扁平化
keywords_series = keywords_series.explode()

# 去除空值
keywords_series = keywords_series.replace('', np.nan).dropna()

# 统计各个词条的出现次数
word_count = keywords_series.value_counts()

# 将结果降序排列
word_count_sorted = word_count.sort_values(ascending=False)
# 使用itertools.combinations()函数获取所有可能的两两组合
combs = list(combinations(keywords_series, 2))

# 将每个组合看作是一条边，构建一个有向图
edges = pd.DataFrame(combs, columns=['source', 'target'])

# 统计每条边的出现次数，即两个关键词在同一篇文献中出现的次数
edge_counts = edges.groupby(['source', 'target']).size().reset_index(name='weight')

# 将关键词映射为数字ID
node_ids = {node: i for i, node in enumerate(set(edges['source']).union(set(edges['target'])))}

# 构建COO稀疏矩阵，表示共现矩阵
row_indices = [node_ids[source] for source, _ in edge_counts[['source', 'target']].itertuples(index=False)]
col_indices = [node_ids[target] for _, target in edge_counts[['source', 'target']].itertuples(index=False)]
co_occurrence_matrix = coo_matrix((edge_counts['weight'], (row_indices, col_indices)))

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer

# 对共现矩阵进行行归一化
co_occurrence_matrix = normalize(co_occurrence_matrix)

# 计算TF-IDF权重
tfidf_transformer = TfidfTransformer(use_idf=True)
tfidf_matrix = tfidf_transformer.fit_transform(co_occurrence_matrix)


# 缩放权重到0-1之间
min_val = np.min(tfidf_matrix.data)
max_val = np.max(tfidf_matrix.data)
tfidf_matrix.data = (tfidf_matrix.data - min_val) / (max_val - min_val)

# 将稀疏矩阵转化为DataFrame对象
df_tfidf = pd.DataFrame(tfidf_matrix.todense(), columns=node_ids.keys())

# 选择出现次数前50多的关键词
top_words = word_count_sorted.head(50).index.tolist()
df_tfidf = df_tfidf[top_words]

# 构建关键词知识图谱，节点为关键词，边为两个关键词之间的tf-idf权重
edges = []
for i, row in df_tfidf.iterrows():
    for j, value in row.iteritems():
        if i != j and value > 0:
            edges.append((i, j, value))
df_graph = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
import networkx as nx

# 构建无向图
G = nx.Graph()

# 添加节点和节点权重
for node in df_tfidf.columns:
    node_weight = np.sum(df_tfidf[node])
    print(f"Node {node}: weight={node_weight}")
    G.add_node(node, weight=node_weight)
    # 对于未赋值的节点，将weight设为0
    if 'weight' not in G.nodes[node]:
        G.nodes[node]['weight'] = 0
    # 如果节点权重不是数值，则将其设为0
    elif not isinstance(G.nodes[node]['weight'], (int, float)):
        G.nodes[node]['weight'] = 0

# 添加边和边权重
for _, row in df_graph.iterrows():
    source = row['source']
    target = row['target']
    weight = row['weight']
    G.add_edge(source, target, weight=weight)

# 筛选边：保留权重前n大的边
n = 20
top_n_edges = df_graph.nlargest(n, 'weight')

# 构建只包含前n大权重边的子图
edge_list = list(zip(top_n_edges['source'], top_n_edges['target']))
H = G.edge_subgraph(edge_list)
import matplotlib.pyplot as plt
# 将图谱导出为gexf格式
nx.write_gexf(H, '../knowledge_graph.gexf')
# 设置节点大小为其权重的平方根
node_sizes = [np.sqrt(G.nodes[node]['weight']) * 200 for node in G.nodes]

# 绘制图谱
pos = nx.spring_layout(H, k=-500, iterations=50)
nx.draw_networkx_nodes(H, pos, node_color='lightblue', node_size=node_sizes)
nx.draw_networkx_edges(H, pos, edge_color='gray', alpha=0.5, max_dist=300)
nx.draw_networkx_labels(H, pos, font_size=10)
plt.axis('off')
plt.show()



