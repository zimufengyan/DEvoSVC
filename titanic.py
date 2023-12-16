# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      titanic
   Author :         zmfy
   DateTime :       2023/12/15 18:57
   Description :    
-------------------------------------------------
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from svc_utils import DEvoSVC

print("Loading data and Preprocessing...")
train_data = pd.read_csv("./data/train.csv")

# 通过拟合数据的方式构造缺省的Age数据
# 提取所有数值型信息进行后续随机森林拟合
df = train_data.copy(deep=True)  # 先拷贝一份


def set_missing_age(df):
    age_df = df[["Age", "Parch", "SibSp", "Pclass", "Fare"]]
    # 乘客分为已知年龄和未知年龄两类
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    # y即为目标年龄
    y = known_age[:, 0]
    # x即为目标属性值
    x = known_age[:, 1:]

    # 拟合RandomForestRegressor
    rft = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rft.fit(x, y)

    # 预测
    pre_ages = rft.predict(unknown_age[:, 1:])
    # 将预测数据加入到已有数据中
    df.loc[(df.Age.isnull()), "Age"] = pre_ages
    return df


# 处理cabin数据，分为”有“和”无“两类，即为"Yes" or "No"
def set_cabin(df):
    df.loc[(df.Cabin.notnull()), "Cabin"] = "Yes"
    df.loc[(df.Cabin.isnull()), "Cabin"] = "No"
    return df


# 将文本型数据进行编码，使用pandas.get_dummies()，并拼接到df上
def encode_df(df):
    dummies_Cabin = pd.get_dummies(data=df.Cabin, prefix="Cabin")
    dummies_Embarked = pd.get_dummies(data=df["Embarked"], prefix="Embarked")
    dummies_Sex = pd.get_dummies(data=df["Sex"], prefix="Sex")
    dummies_Pclass = pd.get_dummies(data=df["Pclass"], prefix="Pclass")

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Pclass, dummies_Sex], axis=1)
    df.drop(["Pclass", "Sex", "Embarked", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
    return df


def scale_age_fare(df):
    scaler = StandardScaler()
    age_df = pd.DataFrame({"Age": df.Age})  # 转换为二维数组
    age_scaler = scaler.fit(age_df)
    df["Age_Scaled"] = scaler.fit_transform(age_df, age_scaler)
    fare_df = pd.DataFrame({"Fare": df.Fare})
    fare_sacler = scaler.fit(fare_df)
    df["Fare_Scaled"] = scaler.fit_transform(fare_df, fare_sacler)
    return df


df = set_missing_age(df)
df = set_cabin(df)
df = encode_df(df)
df = scale_age_fare(df)

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Cabin.*|Embarked.*|Pclass.*|Sex.*|Fare_.*')
train_np = train_df.values

# 将年龄Age离散化， Age <= 12 定义为child, Age > 60 定义为elderly, 中间定义为adult
new_train_df = df.copy(deep=True)
new_train_df["Child"] = np.where(new_train_df.Age <= 12, 1, 0)
new_train_df["Elderly"] = np.where(new_train_df.Age > 60, 1, 0)

# 猜测原数据Name中带有Mrs且Parch>1的为母亲，新增Mothe字段，其获救情况应该要大一些
new_train_df["Mother"] = np.where(
    (train_data.Name.apply(lambda y: True if re.search(r'Mrs', y) else False)) & train_data.Parch > 0, 1, 0)

# 将Parch, SibSp和自己加起来组成新特征Family_Size
new_train_df["Family_Size"] = new_train_df.SibSp + new_train_df.Parch + 1

# 将Cabin中的数字提取出来，如果没有则设为0,并进行归一化
new_train_df["Cabin_Num"] = train_data.Cabin.apply(
    lambda y: int(re.search(r'\D*(\d+)', str(y)).group(1)) if re.search(r'\D*(\d+)', str(y)) else 0)
scaler = MinMaxScaler()
Cabin_Num = pd.DataFrame({"Cabin_Num": new_train_df.Cabin_Num.values})
Cabin_Num_Sacled = scaler.fit_transform(Cabin_Num)
new_train_df["Cabin_Num_Scaled"] = Cabin_Num_Sacled

# 选出特征值
new_train = new_train_df.filter(regex='Survived|Child|Adult|Mother|Elderly|Family_Size|Fare_.*|Cabin_Num_Scaled|Sex_'
                                      '.*|Pclass_.*')

X, y = new_train.values[:, 1:], new_train.values[:, 0]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=520)
print(f"训练集数据量: {X_train.shape}, {y_train.shape}; 测试集数据量: {X_val.shape}, {y_val.shape}")

print("Training...")
svc = DEvoSVC(C=2.5, max_iters=300, mutation_type="de/best/1/bin")
svc.fit(X_train, y_train)
y_pred1 = svc.predict(X_val)
y_pred1 = np.squeeze(y_pred1, axis=1)

sk_svc = SVC(C=2.5)
sk_svc.fit(X_train, y_train)
y_pred2 = sk_svc.predict(X_val)

y_test = np.array(y_val)
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

plt.figure(figsize=(8, 6), dpi=100)
plt.plot(svc.optimizer.best_y_per_iter)
plt.ylabel("object fitness value")
plt.xlabel("iteration")
plt.savefig("Titanic-Optimization.png")