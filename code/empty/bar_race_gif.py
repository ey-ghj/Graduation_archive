import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 读取CSV文件并合并成一个DataFrame对象
# 读取第一个CSV文件
df1 = pd.read_csv('../source1.csv', encoding='gbk')

# 读取第二个CSV文件
df2 = pd.read_csv('../source2.csv', encoding='gbk')
merged_df = pd.concat([df1, df2], ignore_index=True)

# 按年份循环生成每一帧动画
frames = []
for year in range(2019, 2023):
    year_df = merged_df[merged_df['Year-年'] == year]
    keywords = year_df['Keyword-关键词'].str.split(';').explode()
    keyword_count = keywords.value_counts()
    top_keywords = keyword_count[:10].sort_values(ascending=False)

    frames.append(go.Frame(
        data=[go.Bar(x=top_keywords,
                     y=top_keywords.index,
                     marker=dict(color='rgba(50, 171, 96, 0.6)',
                                 line=dict(color='rgba(50, 171, 96, 1.0)', width=1)),
                     orientation='h')],
        name=str(year)
    ))

# 创建子图布局和初始数据
fig = make_subplots(rows=1, cols=1)
initial_year = 2019
initial_df = merged_df[merged_df['Year-年'] == initial_year]
initial_keywords = initial_df['Keyword-关键词'].str.split(';').explode()
initial_keyword_count = initial_keywords.value_counts()
initial_top_keywords = initial_keyword_count[:10].sort_values(ascending=False)
fig.add_trace(go.Bar(x=initial_top_keywords,
                     y=initial_top_keywords.index,
                     marker=dict(color='rgba(50, 171, 96, 0.6)',
                                 line=dict(color='rgba(50, 171, 96, 1.0)', width=1)),
                     orientation='h'), 1, 1)

# 设置布局和动画参数
fig.update_layout(
    title_text="关键词条形竞赛",
    xaxis_title="出现次数",
    yaxis_title="关键词",
    font=dict(size=12),
    margin=dict(l=100, r=20, t=50, b=20),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[dict(label="播放",
                      method="animate",
                      args=[None, {"frame": {"duration": 500, "redraw": False},
                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                 dict(label="暂停",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate", "transition": {"duration": 0}}])]
    )],
    # 设置动画帧参数
    frames=frames
)

# 显示动图
fig.show()
