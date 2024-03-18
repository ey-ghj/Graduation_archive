#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 17:21
# @Author  : Cc
# @Site    : 
# @Version : 
# @File    : keywords_kg.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.sparse import coo_matrix
import time
# 记录开始时间
start_time = time.time()
# 读取两个CSV文件，合并成一个DataFrame对象
df1 = pd.read_csv('source1.csv', encoding='gbk')
df2 = pd.read_csv('source2.csv', encoding='gbk')
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
import networkx as nx

# 获取前50个关键词及其对应的TF-IDF权重
top_keywords = df_tfidf.sum().sort_values(ascending=False)[:50]

# 输出结果
print(top_keywords)

# 将共现矩阵转化为无向图
graph = nx.Graph(co_occurrence_matrix)

# 数据清洗和转换
keywords_series = merged_df['Keyword-关键词'].str.replace('[\n\t\s]+', ' ', regex=True)  # 替换非法字符
keywords_series = keywords_series.str.lower()  # 转化为小写字母
keywords_series = keywords_series.str.split(';').explode().replace('', np.nan).dropna()  # 按分号拆分，并扁平化
# 创建一个空图
graph = nx.Graph()

# 将所有节点加入到图中
for node in set(keywords_series):
    graph.add_node(node)

# 将所有边加入到图中，并将TF-IDF权重作为边权重
for u, v, weight in zip(row_indices, col_indices, tfidf_matrix.data):
    source = list(node_ids.keys())[list(node_ids.values()).index(u)]
    target = list(node_ids.keys())[list(node_ids.values()).index(v)]
    graph.add_edge(source, target, weight=weight)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial']
# 计算节点的PageRank值
pr = nx.pagerank(graph)

# 将节点PageRank值作为节点大小
node_size = [3000 * pr[node] for node in graph.nodes()]
# 将图可视化
pos = nx.spring_layout(graph, seed=42)  # 使用Spring Layout布局算法
nx.draw_networkx_nodes(graph, pos, node_size=node_size, alpha=0.8, cmap=plt.cm.Blues)
nx.draw_networkx_edges(graph, pos, alpha=0.4, edge_color='grey')
nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif', font_color='black')
plt.axis('off')
plt.savefig("./output/keywords_kg.png", dpi=300)
nx.write_gexf(graph, "./output/keywords_graph.gexf")
plt.show()

# 选择前50个关键词构建子图
top_keywords = df_tfidf.sum().sort_values(ascending=False)[:50]
subgraph_nodes = list(top_keywords.index)
subgraph = graph.subgraph(subgraph_nodes)

# 计算节点的PageRank值
pr = nx.pagerank(subgraph)

# 将节点PageRank值作为节点大小
node_size = [3000 * pr[node] for node in subgraph.nodes()]

# 将图可视化
pos = nx.spring_layout(subgraph, seed=42)  # 使用Spring Layout布局算法
nx.draw_networkx_nodes(subgraph, pos, node_size=node_size, alpha=0.8, cmap=plt.cm.Blues)
nx.draw_networkx_edges(subgraph, pos, alpha=0.4, edge_color='grey')
nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif', font_color='black')
plt.axis('off')
plt.savefig("./output/keywords50_kg.png", dpi=300)
nx.write_gexf(subgraph, "./output/keywords50_graph.gexf")
plt.show()
# 记录结束时间
end_time = time.time()

# 输出函数执行时间
print("Time elapsed: ", end_time - start_time, "seconds")