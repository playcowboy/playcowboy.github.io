import pandas as pd  # 导入pandas库，用于数据处理
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于数据集拆分
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.metrics import mean_squared_error  # 导入均方误差计算函数
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化

# 1. 加载房价数据集
data = pd.read_csv("F:\\boston.csv")  # 读取房价数据集文件

# 2. 准备特征和目标变量
X = data.drop('MEDV', axis=1)  # 将特征列名替换为实际的特征列名
y = data['MEDV']  # 将目标变量列名替换为实际的目标变量列名=

# 3. 数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 将数据集拆分为训练集和测试集，其中测试集占比为20%

# 4. 创建线性回归模型并拟合数据
model_1015 = LinearRegression()  # 创建线性回归模型对象
model_1015.fit(X_train, y_train)  # 使用训练集数据拟合线性回归模型

# 5. 预测房价
y_train_pred = model_1015.predict(X_train)  # 对训练集数据进行预测
y_test_pred = model_1015.predict(X_test)  # 对测试集数据进行预测

# 6. 模型评估
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)  # 计算训练集的均方根误差
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)  # 计算测试集的均方根误差

print(f"训练集均方根误差：{train_rmse}")  # 输出训练集的均方根误差
print(f"测试集均方根误差：{test_rmse}")  # 输出测试集的均方根误差

# 7. 结果可视化（以实际值与预测值散点图为例）
plt.scatter(y_test, y_test_pred)  # 绘制实际值与预测值的散点图
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 绘制一条参考线，表示预测值与实际值之间的差距
plt.xlabel('实际房价')  # 设置x轴标签为“实际房价”
plt.ylabel('预测房价')  # 设置y轴标签为“预测房价”
plt.title('房价预测结果')  # 设置图表标题为“房价预测结果”
plt.show()  # 显示图表