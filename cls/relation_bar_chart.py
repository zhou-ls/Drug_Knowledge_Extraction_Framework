# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from utils import re_excel, bar_chart

# 读取EXCEL数据
df = pd.read_excel(re_excel)
label_list = list(df['关系'].value_counts().index)
num_list = df['关系'].value_counts().tolist()

# Mac系统设置中文字体支持
# plt.rcParams["font.family"] = 'Arial Unicode MS'
# Windows系统设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']

# 利用Matplotlib模块绘制条形图
x = range(len(num_list))
rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
# plt.ylim(0, 800) # y轴范围
plt.ylabel("数量")
plt.xticks([index + 0.1 for index in x], label_list)
plt.xticks(rotation=45)     # x轴的标签旋转45度
plt.xlabel("有无相互作用")
plt.title("关系频数统计")
plt.legend()

# 条形图的文字说明
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")

# plt.show()
plt.savefig(bar_chart)
