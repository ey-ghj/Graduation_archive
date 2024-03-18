import pandas as pd
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
keywords_series = keywords_series.replace('', pd.NaT).dropna()

# 将关键词映射到数字索引
keywords, keyword_indices = pd.factorize(keywords_series)

# 统计关键词出现次数
word_count = pd.Series(keywords).value_counts()

# 构建关键词对，并统计它们在所有文献中出现的次数
pairs = []
co_counts = []
for kw in keywords_series:
    if len(kw) > 1:
        indices = keyword_indices[kw]
        pairs += list(combinations(indices, 2))
        co_counts += [1] * (len(kw) * (len(kw) - 1) // 2)

# 将关键词对和共现次数转换成稀疏矩阵
matrix = coo_matrix((co_counts, zip(*pairs)), shape=(word_count.size, word_count.size))

# 将稀疏矩阵转换成密集矩阵
dense_matrix = matrix.toarray()

# 打印共现矩阵
print("共现矩阵：")
print(dense_matrix)
