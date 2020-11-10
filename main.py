# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import my_knn as mk
import matplotlib.pyplot as plt


def gen_data(mean, cov, n = 50):
    """
    mean:多维分布的均值,维度为1
    cov:协方差矩阵,必须是对称的且需为半正定矩阵
    n:产生的样本量
    """
    data = np.random.multivariate_normal(mean, cov, n, "raise")
    return np.round(data, 3)


#生成数据例子

#data01 = gen_data([2, 2], [[1, 0], [0, 1]], 10)
#
##print(data01)

    
#生成数据
data01 = gen_data([2, 3], [[1, 0], [0, 1]], 50)
data01 = pd.DataFrame(columns=['x','y'], data = data01)
data01['c'] = 1
data02 = gen_data([6, 7], [[2, 0], [0, 3]], 70)
data02 = pd.DataFrame(columns=['x','y'], data = data02)
data02['c'] = 2

data = data01.append(data02)
#print(data.index)


#绘图
#plt.scatter(data["x"], data["y"], c = data["c"])
#plt.show()

#计算样本间距离
dis_dict = {}
knn_use_data = data[["x", "y"]]

#print(knn_use_data)
for i in knn_use_data.index:
    dis_dict[i] = []
    data_i = list(knn_use_data.iloc[i])
    for j in data.index:
        data_j = list(knn_use_data.iloc[j])
#        print(data_j)
        dis_dict[i].append(mk.distance(data_i, data_j))


#获取距离目标点最近的几个点的index
point_index = {}

for item in dis_dict:
    point_index[item] = mk.get_index(dis_dict[item], k = 3)

#print(point_index)


#获取共现概率
pij_list = {}

for i in point_index:
    pij_list[i] = []
    for j in point_index:
        pij_list[i].append(mk.co_occ_pro(point_index[i], point_index[j]))

#print(pij_list)


#转换为矩阵形式
pij_matrix = [pij_list[i] for i in pij_list]
#print(pij_matrix)


#绘制热力图

import seaborn as sns
sns.heatmap(pij_matrix, cmap = "YlGnBu")
