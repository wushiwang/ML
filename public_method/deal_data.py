from numpy import ravel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


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