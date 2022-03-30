import os

from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.python.keras.utils import np_utils


def resize_images(path1, path2):
    """设置图片大小"""
    images = os.listdir(path1)
    img_rows, img_cols = 224, 224
    for image in images:
        im = Image.open(path1 + '\\' + image)
        img = im.resize((img_rows, img_cols))
        gray = img.convert('L')
        gray.save(path2 + '\\' + image, 'JPEG')


def get_images(path2):
    images = os.listdir(path2)
    img_matrix = np.array([np.array(Image.open(path2 + '\\' + img)).flatten() for img in images], 'f')
    return img_matrix


def transform_array(img_matrix):
    lists = []
    for i in range(len(img_matrix)):
        lists.append(img_matrix[i])
    return lists


if __name__ == '__main__':
    # 数据预处理
    path1 = 'G:\\web_devment\\ML\\knee\\origin_data'
    path2 = 'G:\\web_devment\\ML\\knee\\user_data'
    resize_images(path1, path2)
    img_mat = get_images(path2)
    li = np.array(transform_array(img_mat))
    num_samples = len(li)

    img_rows, img_cols = 224, 224
    img_channels = 1
    nb_classes = 2

    # 制作数据集
    label = np.ones((num_samples,), dtype=int)
    label[0:69] = 0
    label[69:171] = 1
    data, label = shuffle(li, label, random_state=4)
    train_data = [data, label]

    # 划分数据集
    (X, y) = (train_data[0], train_data[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4)

    # 归一化数据集
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')

    X_train /= 225
    X_test /= 225
    X_val /= 225

    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    X /= 225

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    y = np_utils.to_categorical(y, nb_classes)
    class_names = ['non_injured', 'injured']

    # 训练参数
    batch_size = 32
    np_epoch = 2
    nb_pool = 2
    nb_conv = 3
    nb_filters1 = 32
    nb_filters2 = 64

    # 未提升提前终止
    earlystopper1 = EarlyStopping(monitor='loss', patience=10, verbose=1)

    # 保存训练期间最好模型
    checkpointer1 = ModelCheckpoint('best_model1.h5', monitor='val_accuracy', verbose=1, save_best_only=True,
                                    save_weights_only=True)

    model2 = Sequential()
    model2.add(Conv2D(8, kernel_size=3, input_shape=(img_rows, img_cols, 1), activation='relu', padding='same'))
    model2.add(MaxPool2D(2))
    model2.add(Conv2D(8, kernel_size=3, activation='relu', padding='same'))
    model2.add(Dropout(0.5))
    model2.add(MaxPool2D(2))

    model2.add(Flatten())
    model2.add(Dense(34, activation='relu'))
    model2.add(Dense(2, activation='sigmoid'))
    model2.add(Dropout(0.25))

    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    training2 = model2.fit(X_train, Y_train
                           , batch_size=batch_size
                           , epochs=100
                           , validation_data=(X_val, Y_val)
                           , callbacks=[checkpointer1])

    test_loss, test_acc = model2.evaluate(X_test, Y_test)
    print('Test accuracy:', test_acc)
