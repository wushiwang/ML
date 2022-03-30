import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

shop_train = pd.read_csv('./data/shop.csv')
user_train = pd.read_csv('./data/user.csv')
test = pd.read_csv('./data/test.csv')

print(shop_train.head(5))

shop_mall = shop_train[['shop_id', 'mall_id']]
shop_category = shop_train[['shop_id', 'category_id']]
user_train = pd.merge(user_train, shop_mall, on=['shop_id'], how='left')
user_train = pd.merge(user_train, shop_category, on=['shop_id'], how='left')

train_test = user_train.append(test)


def std_list(x):
    result = []
    biggest = 0
    smallest = -200
    for each in x:
        fen_zi = int(each) - int(smallest)
        fen_mu = int(biggest)
        result.append(fen_zi / fen_mu)

    return result


for name, group in train_test.groupby('mall_id'):
    print(name)
    wifi_info = group.wifi_infos.tolist()
    shop_id = group.shop_id.tolist()
    mall_id = group.mall_id.tolist()

    row_id = group.row_id.tolist()
    user_id = group.user_id.tolist()

    print(len(shop_id))
    feature_wifi = []
    for i in range(0, len(shop_id)):
        wifi = wifi_info[i].split(';')
        wifi_name = []
        wifi_strength = []
        for each in wifi:
            wifi_name.append(each.split('|')[0])
            wifi_strength.append(each.split('|')[1])

        wifi_strength = std_list(wifi_strength)
        dic = {}
        dic = {wifi_name[num]: wifi_strength[num] for num in range(0, len(wifi_name))}

        dic['shop_id'] = shop_id[i]
        dic['mall_id'] = mall_id[i]
        dic['row_id'] = row_id[i]
        dic['user_id'] = user_id[i]
        feature_wifi.append(dic)

    df = pd.DataFrame(feature_wifi)
    name = mall_id[0]
    path = ''
    df.to_csv(path, index=False)
    print('mall_id为' + mall_id[0])
