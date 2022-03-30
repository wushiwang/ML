import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor as xgbr
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer

train_url = r'..\user_data\used_car_train.csv'
test_url = r'..\user_data\used_car_test.csv'
output_url = r'..\prediction_result\predictions.csv'


# 加载数据集
def load_data(train_url, test_url):
    train_data = pd.read_csv(train_url, delimiter=',')
    test_data = pd.read_csv(test_url, delimiter=',')
    return train_data, test_data


# 交叉验证
def cross_verify(model, X, Y):
    scores = cross_val_score(model, X, Y, verbose=1, cv=5, n_jobs=4, scoring=mean_absolute_error)
    scores = pd.DataFrame(scores.reshape(1, -1))
    scores.columns = ['cv' + str(i) for i in range(1, 6)]
    scores.index = ['MAE']
    print(scores)


def build_model_lr(x_train, y_train):
    reg_model = LinearRegression()
    reg_model.fit(x_train, y_train)
    return reg_model


def build_model_svr(x_train, y_train):
    svr_model = SVR(kernel='rbf')
    svr_model.fit(x_train, y_train)
    return svr_model


if __name__ == "__main__":
    train_data, test_data = load_data(train_url, test_url)
    price = train_data['price']
    train_data.drop(['price'], axis=1, inplace=True)
    train_data.drop(['SaleID'], axis=1, inplace=True)

    # SaleID不作为训练特征
    SaleID = test_data['SaleID']
    test_data.drop(['SaleID'], axis=1, inplace=True)

    # param_grid = {'learning_rate':[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    #               'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #               'min_child_weight':[1, 2, 3, 4, 5, 6],
    #               'n_estimators':[600, 650, 700, 750],
    #               'gamma':[1, 2, 3, 4, 5, 6, 7, 8, 9],
    #               'subsample':np.linspace(0.5, 1.5, 10),
    #               'colsample_bytree':np.linspace(0.5, 1.5, 10),
    #               'reg_alpha':[0, 0.05, 0.1, 1, 2, 3],
    #               'reg_lambda':[0.1, 0.5, 1, 1.5, 2, 3]}

    # 确定n_estimators
    # param_grid = {'n_estimators':[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 750, 800, 850, 900, 950, 1000]} # 1000
    # # 确定max_depth,min_child_weight
    # param_grid = {'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # 10
    #               'min_child_weight':[1, 2, 3, 4, 5, 6]} # 2
    # # 确定gamma
    # param_grid = {'gamma':[1, 2, 3, 4, 5, 6, 7, 8, 9]} # 1
    # param_grid = {'gamma':np.linspace(0.1, 1, 10)} # 0.7
    # # 确定subsample,colsample_bytree
    # param_grid = {'subsample':np.linspace(0.1, 1, 10), # 0.8
    #               'colsample_bytree':np.linspace(0.1, 1, 10)} # 0.9
    # # 确定reg_alpha,reg_lambda
    # param_grid = {'reg_alpha':[0, 0.05, 0.1, 1, 2, 3],
    #               'reg_lambda':[0.1, 0.5, 1, 1.5, 2, 3]}
    # 确定learning_rate
    # param_grid = {'learning_rate':[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]}
    #
    # print('search best params ...')
    # start = time.time()
    # grid_search = GridSearchCV(reg, param_grid, scoring='neg_mean_absolute_error', iid=False, cv=5)
    # grid_search.fit(train_data, price)
    # evalute_result = grid_search.cv_results_
    # print('每轮的迭代结果:{}'.format(evalute_result))
    # print('best_params:', grid_search.best_params_)
    # print('best_score:', grid_search.best_score_)
    # print('GridSearchCV process use %.2f seconds'%(time.time()-start))

    # 模型定义
    print('xgb模型训练中...')
    model_xgb = xgbr(learning_rate=0.1,  # 0.1, default=0.3
                     n_estimators=1000,  # 1000
                     max_depth=10,  # 10, default=6
                     min_child_weight=2,  # 2, default=1
                     subsample=0.8,  # 0.8, default=1
                     colsample_bytree=0.9,  # 0.9, default=1
                     gamma=0.7,  # 0.7, default=0
                     reg_alpha=0,  # 0, default=0
                     reg_lambda=0.1,  # 0.1, default=1
                     n_jobs=8)
    model_xgb.fit(train_data, price)
    y_pred_xgb = model_xgb.predict(test_data)
    # y_pred_xgb = np.round(np.exp(y_pred_xgb), 0)

    print('lgb模型训练中...')
    model_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.1, n_jobs=8)
    model_lgb.fit(train_data, price)
    y_pred_lgb = model_lgb.predict(test_data)
    # y_pred_lgb = np.round(np.exp(y_pred_lgb), 0)

    print('gbdt模型训练中...')
    model_gbdt = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
    model_gbdt.fit(train_data, price)
    test_data
    y_pred_gbdt = model_gbdt.predict(test_data)
    # y_pred_gbdt = np.round(np.exp(y_pred_gbdt), 0)

    # 交叉验证 (参数置为default)
    # cross_verify(reg, train_data, price)

    # 模型融合
    print('模型融合中...')
    svr_rbf = SVR(kernel='rbf')
    ## 第一层
    train_xgb_pred = model_xgb.predict(train_data)
    train_lgb_pred = model_lgb.predict(train_data)
    train_gbdt_pred = model_gbdt.predict(train_data)

    Stark_X_train = pd.DataFrame()
    Stark_X_train['Method_1'] = train_xgb_pred
    Stark_X_train['Method_2'] = train_lgb_pred
    Stark_X_train['Method_3'] = train_gbdt_pred

    Stark_X_test = pd.DataFrame()
    Stark_X_test['Method_1'] = y_pred_xgb
    Stark_X_test['Method_2'] = y_pred_lgb
    Stark_X_test['Method_3'] = y_pred_gbdt
    ## 第二层
    ## 训练集测试
    print('模型测试中...')
    # model_lr_Stacking = build_model_lr(Stark_X_train, price) # lr作为最终的meta-regressor
    # train_lr_pre_Stacking = model_lr_Stacking.predict(Stark_X_train)
    model_svr_Stacking = build_model_svr(Stark_X_train, price)  # svr作为最终的meta-regressor
    # train_svr_pre_Stacking = model_svr_Stacking.predict(Stark_X_train)
    # price = np.round(np.exp(price), 0)
    # train_lr_pre_Stacking = np.round(np.exp(train_lr_pre_Stacking), 0)
    # print('MAE of Stacking-LR:', mean_absolute_error(price, train_lr_pre_Stacking))
    # train_svr_pre_Stacking = np.round(np.exp(train_svr_pre_Stacking), 0)
    # print('MAE of Stacking-SVR:', mean_absolute_error(price, train_svr_pre_Stacking))

    ## 测试集
    # print('Predict Stacking-LR...')
    # subA_Stacking = model_svr_Stacking.predict(Stark_X_test)
    print('Predict Stacking-SVR...')
    subA_Stacking = model_svr_Stacking.predict(Stark_X_test)

    # subA_Stacking[subA_Stacking < 10] = 10  ## 去除过小的预测值

    # print('模型训练中...')
    # stregr = StackingRegressor(regressors=[model_xgb, model_lgb, model_gbdt], meta_regressor=svr_rbf)
    # stregr.fit(train_data, price)
    # y_pred = stregr.predict(test_data)

    y_pred = np.round(np.exp(subA_Stacking), 0)
    y = pd.DataFrame(y_pred.astype(int))

    del train_data
    del test_data
    gc.collect()

    data = pd.concat([SaleID, y], axis=1)
    data.columns = ['SaleID', 'price']
    print(data)
    data.to_csv(output_url, decimal=',', index=False)
    print('成功生成预测数据文件。')
