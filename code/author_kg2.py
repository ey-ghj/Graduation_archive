#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 19:13
# @Author  : Cc
# @Site    : 
# @Version : 
# @File    : author_kg.py
# @Software: PyCharm
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 读取第一个CSV文件
df1 = pd.read_csv('source1.csv', encoding='gbk')
# 读取第二个CSV文件
df2 = pd.read_csv('source2.csv', encoding='gbk')
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
# 提取作者信息

flat_authors_list = top_authors
# 设置显示参数
Strength = -500
MaxDistance = 300
Isolates = True

# 构建共现矩阵
cv = CountVectorizer(tokenizer=lambda x: x.split(';'), vocabulary=flat_authors_list, lowercase=False)
co_occurrence_matrix = cv.fit_transform(merged_df['Author-作者']).T * cv.fit_transform(merged_df['Author-作者'])

# 计算TF-IDF权重
tfidf = TfidfTransformer()
tfidf_matrix = tfidf.fit_transform(co_occurrence_matrix)

import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial']

# 构建图谱对象
G = nx.Graph()

# 添加节点
for i, author in enumerate(flat_authors_list):
    G.add_node(author)

# 添加边及权重
rows, cols = tfidf_matrix.nonzero()
for row, col in zip(rows, cols):
    weight = tfidf_matrix[row, col]
    G.add_edge(flat_authors_list[row], flat_authors_list[col], weight=weight)

# 删除孤立节点
G.remove_nodes_from(list(nx.isolates(G)))

# 绘制图谱
pos = nx.spring_layout(G, k=Strength, scale=MaxDistance)
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=tfidf_matrix.data*10, edge_color='gray', alpha=0.5)
# nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', with_labels=False)
#nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='red')
#隐藏所有节点的标签
labels = {node: '' for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_weight='bold')
# 手动添加标签
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize=12, fontweight='bold', ha='center', va='center')

plt.axis('off')
plt.savefig("./output/author_kg.png", dpi=300)
nx.write_gexf(G, "./output/author_graph.gexf")
plt.show()
