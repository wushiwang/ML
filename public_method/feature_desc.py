import matplotlib.pyplot as plt
import seaborn as sns


class FeatureDesc:
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