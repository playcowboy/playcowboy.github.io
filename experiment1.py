import numpy as np  # 导入numpy库，用于进行数值计算
from sklearn.datasets import make_classification  # 从sklearn.datasets模块中导入make_classification函数，用于生成随机数据集
from sklearn.metrics import accuracy_score  # 从sklearn.metrics模块中导入accuracy_score函数，用于计算准确率
from sklearn.model_selection import train_test_split  # 从sklearn.model_selection模块中导入train_test_split函数，用于划分训练集和测试集

# 生成随机数据集
X, y = make_classification(n_samples=100, n_features=4, random_state=42)  # 生成一个包含100个样本、4个特征的随机分类数据集

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 将数据集划分为训练集（80%）和测试集（20%）

# 定义欧氏距离计算函数
def euclidean_distance_1015(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))  # 计算两个向量之间的欧氏距离

# 定义曼哈顿距离计算函数
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))  # 计算两个向量之间的曼哈顿距离

# 定义KNN分类器类
class KNNClassifier_1015:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k  # 设置K值
        if distance_metric == 'euclidean':
            self.distance_fn = euclidean_distance_1015  # 选择欧氏距离计算函数
        elif distance_metric == 'manhattan':
            self.distance_fn = manhattan_distance  # 选择曼哈顿距离计算函数

    def fit(self, X, y):
        self.X_train = X  # 保存训练数据
        self.y_train = y  # 保存训练标签

    def predict(self, X):
        y_pred = []  # 初始化预测结果列表
        for x in X:
            distances = [self.distance_fn(x, x_train) for x_train in self.X_train]  # 计算每个训练样本与当前样本之间的距离
            k_indices = np.argsort(distances)[:self.k]  # 找到距离最近的K个训练样本的索引
            k_nearest_labels = [self.y_train[i] for i in k_indices]  # 获取这些训练样本的标签
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)  # 找到出现次数最多的标签作为预测结果
            y_pred.append(most_common)  # 将预测结果添加到列表中
        return np.array(y_pred)  # 返回预测结果数组

# 定义K值的范围
k_values = range(1, 10)

best_k = 0
best_accuracy = 0

for k in k_values:
    # 创建KNN分类器
    knn = KNNClassifier_1015(k=k)
    
    # 在训练数据上训练模型
    knn.fit(X_train, y_train)
    
    # 在测试数据上进行预测
    y_pred = knn.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 更新最优K值和准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("最优K值：", best_k)
print("准确率：", accuracy)
