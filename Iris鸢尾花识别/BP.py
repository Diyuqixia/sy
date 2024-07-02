import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
start=time.process_time()
# 预处理数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = X_train.T  # 调整数据维度
X_test = X_test.T
y_train = y_train.reshape(1, -1)  # 调整标签维度
y_test = y_test.reshape(1, -1)
n, m = X_train.shape
# 参数设置
input_size = n
hidden_size = 5
output_size = 3
learning_rate = 0.01
epochs = 1000
L1 = []
# 初始化参数
np.random.seed(0)
W1 = np.random.randn(hidden_size, input_size) * 0.01                                                           # random.randn函数生成一个大小为(hidden_size, input_size)的随机数组，并将其乘以0.01
b1 = np.zeros((hidden_size, 1))                                                                                # 创建一个hidden_size行1列的0矩阵
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))
L1 = []


# 定义激活函数
def relu(x):
    return np.maximum(0, x)


# 训练模型
for epoch in range(epochs):
    # 前向传播
    Z1 = np.dot(W1, X_train) + b1  # x矩阵*w矩阵
    A1 = relu(Z1)  # 通过relu函数
    Z2 = np.dot(W2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)  # softmax激活函数
    # 计算损失
    logprobs = -np.log(A2[y_train, range(m)])  # 这些预测概率值取对数
    cost = np.sum(logprobs) / m
    L1.append(cost)
    # 反向传播
    dZ2 = A2
    dZ2[y_train, range(m)] -= 1
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(dZ1, X_train.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    # 更新参数，进行梯度下降算法
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print("Epoch {}, cost: {}".format(epoch,cost))
        L1.append(cost)
# 模型评估
Z1 = np.dot(W1, X_test) + b1
A1 = relu(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
predictions = np.argmax(A2, axis=0)
accuracy = np.mean(predictions == y_test)

# 绘制图像
plt.plot(np.arange(len(L1)), L1, label='Training Loss')
plt.xlabel('epochs')  # 横坐标轴标题
plt.ylabel('the change of losses')  # 纵坐标轴标题
plt.show()
print('测试准确率:', accuracy)
end=time.process_time()
print("final is in ",end-start)