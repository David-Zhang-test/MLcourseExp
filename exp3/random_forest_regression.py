from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# 参数
args = Namespace(
    seed=1234,
    data_file="exp3/iris.csv",
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
    df = df.drop(["Unnamed: 0"] , axis=1) 
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
reg = LogisticRegression()
reg.fit(X_train, y_train)

# 预测
pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)

# 正确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))
# 计算其他的模型评估指标
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="weighted")
print ("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision, recall, F1))