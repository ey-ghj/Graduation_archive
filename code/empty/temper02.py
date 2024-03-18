#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 20:17
# @Author  : Cc
# @Site    : 
# @Version : 
# @File    : temper02.py
# @Software: PyCharm
import pandas as pd

df = pd.read_csv('../university_colum_university_colum.csv', encoding='gb2312')

print(df.head()) # 打印前5行数据
