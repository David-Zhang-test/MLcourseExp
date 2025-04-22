from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# args
args = Namespace(
    seed=1234,
    data_file="sample_data.csv",
    num_samples=100,
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
    coef = [2,3,5],
)
np.random.seed(args.seed)


# data generate with noise
def generate_data(num_samples, coef):
    X = np.array(range(num_samples))
    y_true = coef[0]*X**2+coef[1]*X+coef[2]
    noise = np.random.randn(num_samples) * 1000
    y = y_true+ noise
    return X, y

X, y = generate_data(args.num_samples, args.coef)
data = np.vstack([X, y]).T
df = pd.DataFrame(data, columns=['X', 'y'])
df.head()

# plt.title("Generated data")
# plt.scatter(x=df["X"], y=df["y"])
# plt.show()


# split train and test set
X_train, X_test, y_train, y_test = train_test_split(
    df["X"].values.reshape(-1, 1), df["y"], test_size=args.test_size, 
    random_state=args.seed)
print ("X_train:", X_train.shape)
print ("y_train:", y_train.shape)
print ("X_test:", X_test.shape)
print ("y_test:", y_test.shape)

# data standardize
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.values.reshape(-1,1))
standardized_X_train = X_scaler.transform(X_train)
standardized_y_train = y_scaler.transform(y_train.values.reshape(-1,1)).ravel()
standardized_X_test = X_scaler.transform(X_test)
standardized_y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()

print ("mean:", np.mean(standardized_X_train, axis=0), 
       np.mean(standardized_y_train, axis=0)) # mean 应该是 ~0
print ("std:", np.std(standardized_X_train, axis=0), 
       np.std(standardized_y_train, axis=0))   # std 应该是 1

def SGDLinearModel():
    # model and train
    lm = SGDRegressor(loss="squared_error", penalty="l1", max_iter=args.num_epochs)
    lm.fit(X=standardized_X_train, y=standardized_y_train)


    # evaluation
    pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
    pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_



    train_mse = np.mean((y_train - pred_train) ** 2)
    test_mse = np.mean((y_test - pred_test) ** 2)
    print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))
    # 图例大小
    plt.figure(figsize=(15,5))

    # 画出训练数据
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plt.scatter(X_train, y_train, label="y_train")
    plt.plot(X_train, pred_train, color="red", linewidth=1, linestyle="-", label="lm")
    plt.legend(loc='lower right')

    # 画出测试数据
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plt.scatter(X_test, y_test, label="y_test")
    plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")
    plt.legend(loc='lower right')

    # 显示图例
    plt.show()


def PolyModel():
    degree = 2
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train,y_train)




    # evaluation
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)


    train_mse = np.mean((y_train - pred_train) ** 2)
    test_mse = np.mean((y_test - pred_test) ** 2)
    print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse)) 

    # visualization

    X_min, X_max = min(X_train.min(), X_test.min()), max(X_train.max(), X_test.max())
    # 生成用于绘图的密集且排序好的 X 值
    X_plot = np.linspace(X_min, X_max, 200).reshape(-1, 1) # 生成 200 个点
    # 在这些点上进行预测
    y_plot_predict = model.predict(X_plot)

    # 画出训练数据和拟合曲线
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plt.scatter(X_train.ravel(), y_train, label="y_train") # 散点图用原始数据
    plt.plot(X_plot.ravel(), y_plot_predict, color="red", linewidth=2, linestyle="-", label=f"lm (degree {degree})") # 拟合曲线用排序好的数据
    plt.legend(loc='lower right')
    plt.grid(True) # 添加网格线更清晰

    # 画出测试数据和拟合曲线
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plt.scatter(X_test.ravel(), y_test, label="y_test") # 散点图用原始数据
    plt.plot(X_plot.ravel(), y_plot_predict, color="red", linewidth=2, linestyle="-", label=f"lm (degree {degree})") # 拟合曲线用排序好的数据
    plt.legend(loc='lower right')
    plt.grid(True) # 添加网格线更清晰

    plt.tight_layout() # 调整子图布局，避免重叠
    plt.show()

PolyModel()