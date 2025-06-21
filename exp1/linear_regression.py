
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


############ data processing ############
df = pd.read_csv(args.data_file, header=0)
df = df.dropna()

X = df.drop('MEDV', axis=1) # 移除 MEDV 列作为特征
y = df['MEDV'] # MEDV 列作为目标变量
# split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

############ data standardize ############
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.values.reshape(-1,1))
standardized_X_train = X_scaler.transform(X_train)
standardized_y_train = y_scaler.transform(y_train.values.reshape(-1,1)).ravel()
standardized_X_test = X_scaler.transform(X_test)
standardized_y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()



############# model training ############
lm = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3, max_iter=args.num_epochs)
lm.fit(X=standardized_X_train, y=standardized_y_train)


# evaluation
pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_

train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))
print ("train_R2: {0:.2f}, test_R2: {1:.2f}".format(lm.score(standardized_X_train, standardized_y_train), lm.score(standardized_X_test, standardized_y_test)))
coef = lm.coef_ * (y_scaler.scale_/X_scaler.scale_)
intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - np.sum(coef*X_scaler.mean_)
print(coef)
print(intercept)





# lr = LinearRegression().fit(X=standardized_X_train, y=standardized_y_train)

# pred_train_lr = (lr.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
# pred_test_lr = (lr.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
# train_mse = np.mean((y_train - pred_train_lr) ** 2)
# test_mse = np.mean((y_test - pred_test_lr) ** 2)
# print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))
# print("train_R2: {0:.2f}, test_R2: {1:.2f}".format(lr.score(standardized_X_train, standardized_y_train), lr.score(standardized_X_test, standardized_y_test)))
# coef = lr.coef_ * (y_scaler.scale_/X_scaler.scale_)
# intercept = lr.intercept_ * y_scaler.scale_ + y_scaler.mean_ - np.sum(coef*X_scaler.mean_)
# print(coef)
# print(intercept)


# # # 图例大小
# # plt.figure(figsize=(15,5))

# # # 画出训练数据
# # plt.subplot(1, 2, 1)
# # plt.title("Train")
# # plt.scatter(X_train['RM'], y_train, label="y_train")

# # plt.plot(X_train['RM'], pred_train, color="red", linewidth=1, linestyle="-", label="lm")
# # plt.legend(loc='lower right')

# # # 画出测试数据
# # plt.subplot(1, 2, 2)
# # plt.title("Test")
# # plt.scatter(X_test['RM'], y_test, label="y_test")
# # plt.plot(X_test['RM'], pred_test, color="red", linewidth=1, linestyle="-", label="lm")
# # plt.legend(loc='lower right')

# # 图例大小
# plt.figure(figsize=(15,5)) # 调整整个图的大小

# # 画出训练数据
# plt.subplot(1, 2, 1) # 1行2列的第一个子图
# plt.title("Train: Actual vs. Predicted (SGDRegressor)")
# plt.scatter(y_train, pred_train, label="y_train", s=20) # s调整点的大小
# plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color="red", linewidth=1, linestyle="-", label="Ideal Fit (y=x)")
# plt.xlabel("Actual y_train values")
# plt.ylabel("Predicted pred_train_lr values")
# plt.legend(loc='lower right')
# plt.grid(True)


# # 画出测试数据
# plt.subplot(1, 2, 2) # 1行2列的第二个子图
# plt.title("Test: Actual vs. Predicted (SGDRegressor)")
# plt.scatter(y_test, pred_test, label="y_test", s=20) # s调整点的大小
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=1, linestyle="-", label="Ideal Fit (y=x)")
# plt.xlabel("Actual y_test values")
# plt.ylabel("Predicted pred_test_lr values")
# plt.legend(loc='lower right')
# plt.grid(True)

# # 调整子图之间的间距，避免重叠
# plt.tight_layout()

# # 显示图例
# plt.show()