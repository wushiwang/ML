import warnings
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
train_url = r'..\user_data\used_car_train.csv'

# 显示所有列
pd.set_option('display.max_columns', None)


# 载入数据
def load_data(train_url):
    train_data = pd.read_csv(train_url, delimiter=',')
    return train_data


# 筛选特征个数
def choice_feature_nums(data_x, data_y, col_name):
    n = len(col_name)
    dic = {}
    for i in range(3, n + 1):
        rfe = RFE(estimator=LinearRegression, n_features_to_select=i)
        rfe.fit_transform(data_x, data_y)
        dic[i] = rfe.score()
    plt.xlabel('feature_num')
    plt.ylabel('score')
    plt.plot(dic.keys(), dic.values())
    plt.show()
    return dic


if __name__ == "__main__":
    train_data = load_data(train_url)
    train_y = train_data['price']
    train_data.drop(['SaleID'], axis=1, inplace=True)
    train_data.drop(['price'], axis=1, inplace=True)
    col_name = ['name', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage',
                'regionCode', 'v_0', 'v_3', 'v_12', 'usedTime']
    sfs = SFS(LinearRegression(),
              k_features=13,
              forward=True,
              floating=False,
              scoring='r2',
              cv=0)
    train_data = train_data.fillna(0)
    sfs.fit(train_data, train_y)
    print(sfs.k_feature_names_)
    print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
    fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    plt.grid()
    plt.show()
