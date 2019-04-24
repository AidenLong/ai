# -*- coding:utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化三个中心点
centers = [[1, 1], [-1, -1], [1, -1]]
clusters = len(centers)
# 产生3000组二维的数据，中心点意思是三个中心点，标准差为0.7
X, Y = make_blobs(n_samples=300, centers=centers, cluster_std=0.7, random_state=28)

# 构建kmeans算法
k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
t0 = time.time()  # 当前时间
k_means.fit(X)  # 训练模型
km_batch = time.time() - t0  # 使用kmeans训练数据的消耗时间
print("K-Means算法模型训练消耗时间:%.4fs" % km_batch)

# 构建MiniBatchKMeans
batch_size = 100
mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state=28)
t0 = time.time()
mbk.fit(X)
mbk_batch = time.time() - t0
print("Mini Batch K-Means算法模型训练消耗时间:%.4fs" % mbk_batch)

# 预测结果
km_y_hat = k_means.predict(X)
mbkm_y_hat = mbk.predict(X)

##获取聚类中心点并聚类中心点进行排序
k_means_cluster_centers = k_means.cluster_centers_  # 输出kmeans聚类中心点
mbk_means_cluster_centers = mbk.cluster_centers_  # 输出mbk聚类中心点
print("K-Means算法聚类中心点:\ncenter=", k_means_cluster_centers)
print("Mini Batch K-Means算法聚类中心点:\ncenter=", mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers,
                                  mbk_means_cluster_centers)

# 效果评估
score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    metrics.adjusted_mutual_info_score,
    metrics.mutual_info_score
]

for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(Y, km_y_hat)
    print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))

    t0 = time.time()
    mbkm_scores = score_func(Y, mbkm_y_hat)
    print("Mini Batch K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs\n" % (score_func.__name__, mbkm_scores, time.time() - t0))
