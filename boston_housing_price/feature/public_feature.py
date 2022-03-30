import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ravel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


class PreDealData:
    def __init__(self, data):
        self.data = data
        self.columns = self.data.columns

    def analysis_data(self, n_row):
        """描述性统计"""
        print("数据前{}行".format(n_row))
        print(self.data.head(n_row))  # 查看数据形式

        print('=' * 50)
        print("数据列信息展示")
        print(self.data.info())

        print('=' * 50)
        print("数据描述性统计")
        print(self.data.describe())

        print('=' * 50)
        print("数据空值统计")
        print(self.data.isnull().sum())

        print('=' * 50)
        print("数据维度")
        print(self.data.shape)

    def show_variable_distribution(self, bins=30, kde=True):
        """绘制各变量的分布"""
        for inx, column_name in enumerate(self.columns):
            sns.displot(self.data[column_name], bins=bins, kde=kde)
            plt.xlabel(column_name, fontsize=12)
        plt.show()

    def show_corr(self):
        """展示数据相关性与强相关性"""
        data_corr = self.data.corr().abs()
        sns.heatmap(data_corr, annot=True)
        plt.show()
        sns.heatmap(data_corr, mask=data_corr < 0.5, cbar=False)
        plt.show()

        print("展示强相关性的变量")
        threshold = 0.5
        corr_list = []
        size = data_corr.shape[0]
        for i in range(0, size):
            for j in range(i + 1, size):
                if threshold <= data_corr.iloc[i, j] < 1:
                    corr_list.append([data_corr.iloc[i, j], i, j])

        # 排列相关性
        s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))

        # 输出结果
        for v, i, j in s_corr_list:
            sns.pairplot(self.data, height=10, x_vars=self.columns[i], y_vars=self.columns[j])
            plt.title("%s and %s = %.2f" % (self.columns[i], self.columns[j], v))
            plt.show()


class DealData:
    def __init__(self, data):
        self.data = data
        self.columns = self.data.columns

    def reduce_noise_by_threshold(self, column_name, threshold):
        """通过阙值降噪"""
        self.data = self.data[self.data[column_name] < threshold]

    def scale(self, x_train, y_train, x_test, y_test, scale_method):
        """归一化
            MinMaxScaler :最大最小值规范化
            Normalizer :使每条数据各特征值的和为1
            StandardScaler :为使各特征的均值为0，方差为1
        """
        methods = {'MinMaxScaler': MinMaxScaler(),
                   'Normalizer': Normalizer(),
                   'StandardScaler': StandardScaler()}
        ss = methods[scale_method]
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        y_train = ss.fit_transform(y_train.reshape(-1, 1))
        y_test = ss.transform(y_test.reshape(-1, 1))
        return x_train, y_train, x_test, y_test

    def split_data(self, variable, label, random_state=33, test_size=0.2):
        """分割训练与测试数据集"""
        x_train, x_test, y_train, y_test = train_test_split(variable, label,
                                                            random_state=random_state, test_size=test_size)
        return x_train, y_train, x_test, y_test

    def split_label(self, column_name):
        label = ravel(self.data[column_name].values)
        variable = self.data.drop(column_name, axis=1)
        return variable, label




