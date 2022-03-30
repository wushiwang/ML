from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, LassoCV
from sklearn.metrics import r2_score

from boston_housing_price.feature.public_feature import DealData
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../origin_data/boston_housing.csv')
feature = DealData(data)
feature.reduce_noise_by_threshold("MEDV", 50)
variable, label = feature.split_label('MEDV')
x_train, y_train, x_test, y_test = feature.split_data(variable, label)
x_train, y_train, x_test, y_test = feature.scale(x_train, y_train, x_test, y_test, 'StandardScaler')
print(y_train.shape)
# 线性回归
print('------------线性回归---------------')
lr = LinearRegression()
lr.fit(x_train, y_train.ravel())

y_test_pre = lr.predict(x_test)
y_train_pre = lr.predict(x_train)

print('train:', r2_score(y_train, y_train_pre))
print('test:', r2_score(y_test, y_test_pre))

# fig = plt.subplots(figsize=(7, 5))
# plt.hist(y_train - y_train_pre, bins=30, label="Residuals Linear")
# plt.show()

# 随机梯度回归
print('------------随机梯度---------------')
sgdr = SGDRegressor()
sgdr.fit(x_train, y_train.ravel())
y_pre = sgdr.predict(x_test)
print(sgdr.coef_)
print('train:', sgdr.score(x_train, y_train))
print('test:', sgdr.score(x_test, y_test))

# 岭回归
print('------------岭回归---------------')
alphas = [0.01, 0.1, 1, 10, 100]
ridge = RidgeCV(alphas=alphas, store_cv_values=True)
ridge.fit(x_train, y_train.ravel())
Y_train_pre = ridge.predict(x_train)
Y_test_pre = ridge.predict(x_test)
print('train:', r2_score(y_train, Y_train_pre))
print('test:', r2_score(y_test, Y_test_pre))

print('------------Lasso回归---------------')
alphas = [0.01, 0.1, 1, 10, 100]
lasso = LassoCV(alphas=alphas)
lasso.fit(x_train, y_train.ravel())
Y_train_pre = lasso.predict(x_train)
Y_test_pre = lasso.predict(x_test)
print('train:', r2_score(y_train, Y_train_pre))
print('test:', r2_score(y_test, Y_test_pre))
