
import numpy as np
import pandas as pd
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
'''
Note
From the implementation point of view, this is just plain Ordinary
Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
(scipy.optimize.nnls) wrapped as a predictor object.
'''
from sklearn.linear_model import SGDRegressor
'''
Linear model fitted by minimizing a regularized empirical loss with SGD.
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


args = Namespace(
    seed=1234,
    data_file="house_data.csv",
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
)
np.random.seed(args.seed)

df = pd.read_csv(args.data_file, header=0)

df = df.dropna()


X = df.drop('MEDV', axis=1) # 移除 MEDV 列作为特征
y = df['MEDV'] # MEDV 列作为目标变量


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
# print("训练集特征形状:", X_train.shape)
# print("训练集目标形状:", y_train.shape)
# print("测试集特征形状:", X_test.shape)
# print("测试集目标形状:", y_test.shape)
# print("-" * 30)

X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.values.reshape(-1,1))
standardized_X_train = X_scaler.transform(X_train)
standardized_y_train = y_scaler.transform(y_train.values.reshape(-1,1)).ravel()
standardized_X_test = X_scaler.transform(X_test)
standardized_y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()

lm = SGDRegressor(loss="squared_error", penalty="l1", max_iter=args.num_epochs)
lm.fit(X=standardized_X_train, y=standardized_y_train)


# evaluation
pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_



train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))
