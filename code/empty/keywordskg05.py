import pandas as pd
import numpy as np
from itertools import combinations
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx
import matplotlib.pyplot as plt

# 读取两个CSV文件，合并成一个DataFrame对象
df1 = pd.read_csv('../source1.csv', encoding='gbk')
df2 = pd.read_csv('../source2.csv', encoding='gbk')
merged_df = pd.concat([df1, df2])

# 数据清洗和转换
keywords_series = merged_df['Keyword-关键词'].str.replace('[\n\t\s]+', ' ', regex=True)  # 替换非法字符
keywords_series = keywords_series.str.lower()  # 转化为小写字母
keywords_series = keywords_series.str.split(';').explode().replace('', np.nan).dropna()  # 按分号拆分，并扁平化

# 统计各个词条的出现次数，并仅保留前50个出现次数最多的关键词
word_count = keywords_series.value_counts().head(50)

# 将这些关键词从原始的DataFrame中筛选出来，并获取它们在共现矩阵中的ID
selected_keywords = word_count.index.tolist()
node_ids = {node: i for i, node in enumerate(selected_keywords)}

# 使用筛选后的关键词构建共现矩阵
selected_rows = []
selected_cols = []
selected_weights = []
for row, col, weight in zip(row_indices, col_indices, tfidf_matrix.data):
    source = list(node_ids.keys())[list(node_ids.values()).index(row)]
    target = list(node_ids.keys())[list(node_ids.values()).index(col)]
    if source in selected_keywords and target in selected_keywords:
        selected_rows.append(node_ids[source])
        selected_cols.append(node_ids[target])
        selected_weights.append(weight)
co_occurrence_matrix = coo_matrix((selected_weights, (selected_rows, selected_cols)), shape=(len(selected_keywords), len(selected_keywords)))

# 对共现矩阵进行行归一化和TF-IDF计算
co_occurrence_matrix = normalize(co_occurrence_matrix)
tfidf_transformer = TfidfTransformer(use_idf=True)
tfidf_matrix = tfidf_transformer.fit_transform(co_occurrence_matrix)

# 将稀疏矩阵转化为DataFrame对象
df_tfidf = pd.DataFrame(tfidf_matrix.todense(), columns=selected_keywords)

# 创建一个空图，并将所有节点加入到图中
graph = nx.Graph()
for node in selected_keywords:
    graph.add_node(node)

# 将所有边加入到图中，并将TF-IDF权重作为边权重
for u, v, weight in zip(selected_rows, selected_cols, selected_weights):
    source = list(node_ids.keys())[list(node_ids.values()).index(u)]
    target = list(node_ids.keys())[list(node_ids.values()).index(v)]
    graph.add_edge(source, target, weight=weight)

# 计算节点之间的PageRank值
pagerank = nx.pagerank(graph, alpha=0.85, max_iter=1000, tol=1e-6)

# 绘制关键词图谱
pos = nx.spring_layout(graph)
fig, ax = plt.subplots(figsize=(12, 12))
nx.draw_networkx_nodes(graph, pos, alpha=0.8, cmap=plt.cm.Blues)
nx.draw_networkx_edges(graph, pos, alpha=0.4, edge_color='grey')
nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif', font_color='black')
plt.axis('off')
plt.savefig("./output/keywords_kg.png", dpi=300)
nx.write_gexf(graph, "../output/keywords_graph.gexf")
plt.show()