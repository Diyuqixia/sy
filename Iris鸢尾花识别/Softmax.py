from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
start=time.process_time()
#加载并划分数据集
iris_dataset = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], test_size = 0.3, random_state=1)


# 选用所有特征
Soft_x_train = x_train[:, 0:4]

# 进行softmax回归训练
model_softmax = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=1, random_state=1)
model_softmax.fit(Soft_x_train, y_train)

# 模型测试和使用
line_x_test = x_test[:, 0:4]

correct_cnt = 0
for i, j in zip(line_x_test, y_test):
    predict_species = model_softmax.predict([i])[0]
    if j == round(predict_species):
        correct_cnt += 1
    print("真实: %s, 预测: %s" % (j, predict_species))

print("Accuracy: %s" % (correct_cnt / len(line_x_test)))
end=time.process_time()
print("final is in ",end-start)