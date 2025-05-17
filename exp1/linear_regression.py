
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
lm = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-2, max_iter=args.num_epochs)
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





lr = LinearRegression().fit(X=standardized_X_train, y=standardized_y_train)

pred_train_lr = (lr.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
pred_test_lr = (lr.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
train_mse = np.mean((y_train - pred_train_lr) ** 2)
test_mse = np.mean((y_test - pred_test_lr) ** 2)
print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))
print("train_R2: {0:.2f}, test_R2: {1:.2f}".format(lr.score(standardized_X_train, standardized_y_train), lr.score(standardized_X_test, standardized_y_test)))
coef = lr.coef_ * (y_scaler.scale_/X_scaler.scale_)
intercept = lr.intercept_ * y_scaler.scale_ + y_scaler.mean_ - np.sum(coef*X_scaler.mean_)
print(coef)
print(intercept)
