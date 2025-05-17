from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
# 参数
args = Namespace(
    seed=1234,
    data_file="iris.csv",
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
    max_depth=4,
    min_samples_leaf=5,
    n_estimators=10, # 随机森林中包含的决策树个数
)

# 设置随即种子来保证实验结果的可重复性。
np.random.seed(args.seed)
df = pd.read_csv(args.data_file, header=0)


# 预处理
def preprocess(df):
  
    # 删除掉含有空值的行
    df = df.dropna()

    # # 删除基于文本的特征 (我们以后的课程将会学习怎么使用它们)
    # features_to_drop = ["name", "cabin", "ticket"]
    # df = df.drop(features_to_drop, axis=1)

    # # pclass, sex, 和 embarked 是类别变量
    # # 我们将把字符串转化成浮点数，不再是逻辑回归中的编码变量
    # df['sex'] = df['sex'].map( {'female': 0, 'male': 1} ).astype(int)
    # df["embarked"] = df['embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)

    return df

df = preprocess(df)

mask = np.random.rand(len(df)) < args.train_size
train_df = df[mask]
test_df = df[~mask]
print ("Train size: {0}, test size: {1}".format(len(train_df), len(test_df)))



# 分离 X 和 y
X_train = train_df.drop(["Species"], axis=1)
y_train = train_df["Species"]
X_test = test_df.drop(["Species"], axis=1)
y_test = test_df["Species"]


print(y_test.head())




# 初始化随机森林
forest = RandomForestClassifier(
    n_estimators=args.n_estimators, criterion="entropy", 
    max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
forest.fit(X_train, y_train)

# 预测
pred_train = forest.predict(X_train)
pred_test = forest.predict(X_test)

# 正确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))
# 计算其他的模型评估指标
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="weighted")
print ("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision, recall, F1))