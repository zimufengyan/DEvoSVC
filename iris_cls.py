# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      iris_cls
   Author :         zmfy
   DateTime :       2023/12/14 15:51
   Description :    利用已经实现的基于DE差分进化的SVC二分类其对简化的鸢尾花数据集进行分类
-------------------------------------------------
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns

from svc_utils import DEvoSVC


# 加载鸢尾花数据集
print("Loading Iris dataset...")
iris = load_iris()

# 将数据集转换为DataFrame格式
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# 提取两个类别，这里选择山鸢尾和变色鸢尾
iris_binary = iris_df[iris_df['target'] != 2]

# 分离特征和标签
X = iris_binary.drop('target', axis=1)
y = iris_binary['target']

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据预处理 - 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"训练集数据量: {X_train.shape}, {y_train.shape}; 测试集数据量: {X_test.shape}, {y_test.shape}")

svc = DEvoSVC(max_iters=500, mutation_type='de/best/1/bin')
svc.fit(X_train, y_train)

# acc = svc.evaluate(X_test, y_test)
y_pred1 = np.squeeze(svc.predict(X_test), axis=1)

sk_svc = SVC()
sk_svc.fit(X_train, y_train)
y_pred2 = sk_svc.predict(X_test)

y_test = np.array(y_test)
accuracies = [accuracy_score(y_test, y_pred1), accuracy_score(y_test, y_pred2)]
precisions = [precision_score(y_test, y_pred1), precision_score(y_test, y_pred2)]
recalls = [recall_score(y_test, y_pred1), recall_score(y_test, y_pred2)]
f1s = [f1_score(y_test, y_pred1), f1_score(y_test, y_pred2)]

table = pd.DataFrame(columns=['性能指标', '基于DE的SVC', 'sklearn的SVC'])
table.loc[len(table.index)] = ['准确率', *accuracies]
table.loc[len(table.index)] = ['精确率', *precisions]
table.loc[len(table.index)] = ['召回率', *recalls]
table.loc[len(table.index)] = ['F1分数', *f1s]
print(table)

# 对比多种变异操作的优化过程
print("Comparing four mutation types...")
max_iter = 700
mutation_types = ['DE/rand/1/bin', 'DE/best/1/bin', 'DE/rand-to-best/1/bin', 'DE/best/2/bin', 'DE/rand/2/bin']
models = []
his = dict()
his["Object Fitness"] = []
his["Mutation Type"] = []
# his = {mt: [] for mt in mutation_types}
cnt = 5
his["Iteration"] = [i+1 for i in range(max_iter)] * (len(mutation_types) * cnt)
optimization_processes = []
for mt in mutation_types:
    for j in range(cnt):
        model = DEvoSVC(max_iters=max_iter, mutation_type=mt)
        model.fit(X_train, y_train, eval_train=False)
        # optimization_processes.append(model.optimizer.best_y_per_iter)
        # his[mt] += model.optimizer.best_y_per_iter
        his["Object Fitness"] += model.optimizer.best_y_per_iter
        his["Mutation Type"] += [mt] * max_iter
df = pd.DataFrame(his)
print(df.info())

plt.figure(figsize=(8, 6), dpi=100)
p = sns.relplot(
    x="Iteration", y="Object Fitness", data=df, kind='line',
    hue='Mutation Type', palette='Set1'
)
# plt.legend()
# plt.ylabel("object fitness value")
# plt.xlabel("iteration")
p.fig.savefig("performance.png")


