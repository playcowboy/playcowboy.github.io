import numpy as np  # 导入numpy库，用于进行数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘制图形
from sklearn.datasets import make_blobs  # 从sklearn库中导入make_blobs函数，用于生成随机数据集
from sklearn.metrics import silhouette_score  # 从sklearn库中导入silhouette_score函数，用于计算轮廓系数

# 生成随机数据集
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)  # 使用make_blobs函数生成一个包含300个样本、4个中心的随机数据集，每个样本的标准差为0.60

# K-Means算法实现
def kmeans_1015(X, k, max_iters=100):  # 定义kmeans函数，输入参数为数据集X、聚类数k和最大迭代次数max_iters（默认值为100）
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]  # 随机选择k个样本作为初始质心
    for _ in range(max_iters):  # 进行最多max_iters次迭代
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1), axis=-1)  # 计算每个样本到各个质心的距离，并找到距离最近的质心的索引
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])  # 根据最近质心的索引，计算新的质心位置
        if np.all(centroids == new_centroids):  # 如果新旧质心位置相同，则停止迭代
            break
        centroids = new_centroids  # 更新质心位置
    return labels, centroids  # 返回每个样本的聚类标签和质心位置

# 使用K-Means算法进行聚类
k = 4  # 设置聚类数为4
labels, centroids = kmeans_1015(X, k)  # 调用kmeans函数进行聚类，得到每个样本的聚类标签和质心位置

# 计算轮廓系数
score = silhouette_score(X, labels)  # 使用silhouette_score函数计算轮廓系数
print("轮廓系数：", score)  # 打印轮廓系数

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')  # 绘制散点图，横坐标为样本的第一个特征，纵坐标为样本的第二个特征，颜色由聚类标签决定，点的大小为50，颜色映射为'viridis'
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', alpha=0.5)  # 绘制质心，横坐标为质心的横坐标，纵坐标为质心的纵坐标，点的大小为200，颜色为黑色，透明度为0.5
plt.show()  # 显示图形





