from math import sqrt
import numpy as np
#from numpy import argsort



def distance(x1, x2):
    """
    计算点与点之间的欧氏距离
    """
    d = 0
    for i1, i2 in zip(x1, x2):
#        print("i1:{}, i2:{}".format(i1, i2))
        d += (i1 - i2)**2
    return sqrt(d)


def get_index(dis_list, k):
    """
    获取距离目标点最近的k个点的索引
    """
    dis_list = np.array(dis_list)
    sort_index = dis_list.argsort()
    #从小到大排序
    k_index = sort_index[ :k]
    return list(k_index)


def co_occ_pro(index1, index2):
    """
    计算共现概率
    """
    index1 = set(index1)
    index2 = set(index2)
    x_intersection = index1 & index2
    p12 = len(x_intersection) / len(index1)
    return np.round(p12, 3)

