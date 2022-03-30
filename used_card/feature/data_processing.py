import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as plt
import gc

train_url = '../data/used_car_train_20200313.csv'
test_url = '../data/used_car_testB_20200421.csv'
train_output_path = '../user_data/used_car_train.csv'
test_output_path = '../user_data/used_car_test.csv'

pd.set_option('display.max_columns', None)


# 载入数据
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_url, delimiter=' ')
    test_data = pd.read_csv(test_url, delimiter=' ')
    return train_data, test_data


# 删除无关列
def del_columns(data, col_name):
    data.drop(columns=col_name, axis=1, inplace=True)
    return data


# 删除缺失值对应的行
def del_null(data, col_name):
    data.dropna(axis=0, subset=col_name)
    return data


# 删除异常值对应的行
def del_outliers(data, col_name, scale=3):
    # 箱线图
    def box_plot_outliers(data_ser, box_scale):
        # IQR即尺度*(上四分位点-下四分位点)
        IQR = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - IQR  # 计算下边缘
        val_up = data_ser.quantile(0.75) + IQR  # 计算上边缘
        rule_low = (data_ser < val_low)  # 小于下边缘的值
        rule_up = (data_ser > val_up)  # 大于上边缘的值
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    del_index = []
    for name in col_name:
        data_series = data_n[name]
        rule, value = box_plot_outliers(data_series, scale)
        index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
        del_index.extend(index)
        print("{}需要删除: {} 个数据".format(name, len(index)))

    del_index = list(set(del_index))
    data_n.drop(data_n.index[del_index], inplace=True)
    return data_n


# 填充缺失值
def fill_null(data, col_name):
    for name in col_name:
        print('{0}填充前：{1}'.format(name, data[name].isnull().sum()))
        data[name].fillna(ss.mode(data[name])[0][0], inplace=True)
        print('{0}填充后: {1}'.format(name, data[name].isnull().sum()))
    return data


# 正态变换
def type_convert(data):
    data = np.log(np.array(data))
    return data


# 特征构造
def create_feature(data, col_name):
    data['usedTime'] = (pd.to_datetime(data[col_name[0]], format="%Y%m%d", errors='coerce') -
                        pd.to_datetime(data[col_name[1]], format="%Y%m%d", errors='coerce')).dt.days
    print(data['usedTime'].isnull().sum())
    return data


if __name__ == "__main__":
    train_data, test_data = load_data(train_url, test_url)

    # 不平衡数据删除
    col_name = ['name', 'model', 'power', 'regionCode']
    print('删除之前维度:{}'.format(train_data.shape))
    train_data = del_outliers(train_data, col_name)
    print('删除之后维度:{}'.format(train_data.shape))

    # 取出预测列
    price_data = train_data['price']
    train_data.drop(['price'], axis=1, inplace=True)
    f = plt.figure()
    f.add_subplot(2, 1, 1)
    sns.displot(price_data)
    price_data = type_convert(price_data)
    f.add_subplot(2, 1, 2)
    sns.displot(price_data)
    plt.show()

    # 拼接测试数据集和训练集的数据量，方便拆分
    data = pd.concat([train_data, test_data])
    print("训练集的shape:{0}\n测试集的shape:{1}".format(train_data.shape, test_data.shape))
    del train_data
    del test_data
    gc.collect()

    # 转换object数据
    data['notRepairedDamage'].replace('-', 2.0, inplace=True)
    print(data['notRepairedDamage'].value_counts())

    # 删除特征
    del_features = ['model', 'offerType', 'seller', 'v_1', 'v_2', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
                    'v_11', 'v_13', 'v_14']
    data = del_columns(data, del_features)

    # 特征构造
    col_name = ['creatDate', 'regDate']
    data = create_feature(data, col_name)
    data = del_columns(data, ['regDate', 'creatDate'])

    # 缺失值填充
    col_name = ['bodyType', 'fuelType', 'gearbox', 'usedTime']
    data = fill_null(data, col_name)
    print(data.isnull().sum())

    # 切分数据集
    train_data, test_data = data.iloc[:148913, :], data.iloc[148913:, :]
    train_data['price'] = price_data

    del data
    del price_data
    gc.collect()
    print(train_data.shape)
    print(train_data.head())
    print(test_data.shape)
    print(test_data.head())

    # 生成新的训练集和测试集csv文件
    train_data.to_csv(train_output_path, sep=',', index=False, header=True)
    test_data.to_csv(test_output_path, sep=',', index=False, header=True)
    print('特征工程完成。')
