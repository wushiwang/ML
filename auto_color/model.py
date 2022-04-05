import os

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.backend import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.initializers.initializers_v1 import TruncatedNormal
from keras.layers import Conv2D, RepeatVector, Reshape, UpSampling2D
from keras.optimizer_v1 import RMSprop
from keras.utils.data_utils import threadsafe_generator
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import Input
from keras.models import Model
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import get_default_graph


def encode_model():
    tn = TruncatedNormal(mean=0.0, stddev=0.05)
    # Encoder
    encoder_input = Input(shape=(256, 256, 1))

    encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, bias_initializer=tn)(encoder_input)

    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=tn)(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, bias_initializer=tn)(encoder_output)

    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=tn)(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2, bias_initializer=tn)(encoder_output)

    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=tn)(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same', bias_initializer=tn)(encoder_output)

    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=tn)(encoder_output)

    # Fusion
    embed_input = Input(shape=(1000,))
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same', bias_initializer=tn)(fusion_output)

    # Decoder
    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=tn)(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=tn)(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same', bias_initializer=tn)(decoder_output)
    decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same', bias_initializer=tn)(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same', bias_initializer=tn)(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

    return model


def batch_apply(ndarray, func, *args, **kwargs):
    batch = []
    for sample in ndarray:
        batch.append(func(sample, *args, **kwargs))
    return np.array(batch)


def create_inception_embedding(gray_scaled_rgb):
    inception = InceptionResNetV2(weights='imagenet', include_top=True)
    inception.graph = get_default_graph()

    with inception.graph.as_default():
        embed = inception.predict(gray_scaled_rgb)

    return embed


def process_images(rgb, input_size=(256, 256, 3), embed_size=(299, 299, 3)):
    gray = gray2rgb(rgb2gray(rgb))
    gray = batch_apply(gray, resize, embed_size, mode='constant')
    gray = gray * 2 - 1
    embed = create_inception_embedding(gray)

    re_batch = batch_apply(rgb, resize, input_size, mode='constant')
    re_batch = batch_apply(re_batch, rgb2lab)

    x_batch = re_batch[:, :, :, 0]
    x_batch = x_batch / 50 - 1
    x_batch = x_batch.reshape(x_batch.shape + (1,))

    y_batch = re_batch[:, :, :, 1:]
    y_batch = y_batch / 128

    return [x_batch, embed], y_batch


@threadsafe_generator
def image_a_b_gen(images, batch_size):
    data_gen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)
    while True:
        for batch in data_gen.flow(images, batch_size=batch_size):
            yield process_images(batch)


def get_images(DATASET, filelist, transform_size=(299, 299, 3)):
    """Reads JPEG filelist from DATASET and returns float represtation of RGB [0.0, 1.0]"""
    img_list = []
    for filename in filelist:
        # Loads JPEG image and converts it to numpy float array.
        image_in = img_to_array(load_img(DATASET + filename))

        # [0.0, 255.0] => [0.0, 1.0]
        image_in = image_in / 255

        if transform_size is not None:
            image_in = resize(image_in, transform_size, mode='reflect')

        img_list.append(image_in)
    img_list = np.array(img_list)

    return img_list


def train(model, training_files, batch_size=100, epochs=500, steps_per_epoch=50):
    """Trains the model"""
    training_set = get_images(DATASET, training_files)
    train_size = int(len(training_set) * 0.85)
    train_images = training_set[:train_size]
    val_images = training_set[train_size:]
    val_steps = (len(val_images) // batch_size)
    print("Training samples:", train_size, "Validation samples:", len(val_images))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, cooldown=0, verbose=1, min_lr=1e-8),
        ModelCheckpoint(monitor='val_loss', filepath='colorize.hdf5', verbose=1,
                        save_best_only=True, save_weights_only=True, mode='auto'),
        TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=20, write_graph=True, write_grads=True,
                    write_images=False, embeddings_freq=0)
    ]

    model.fit_generator(image_a_b_gen(train_images, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch,
                        verbose=1, callbacks=callbacks, validation_data=process_images(val_images))


def test(model, testing_files, save_actual=False, save_gray=False):
    test_images = get_images(DATASET, testing_files)
    model.load_weights(filepath='colorize.hdf5')

    print('Preprocessing Images')
    X_test, Y_test = process_images(test_images)

    print('Predicting')
    # Test model
    output = model.predict(X_test)

    # Rescale a*b* back. [-1.0, 1.0] => [-128.0, 128.0]
    output = output * 128
    Y_test = Y_test * 128

    # Output colorizations
    for i in range(len(output)):
        name = testing_files[i].split(".")[0]
        print('Saving ' + str(i) + "th image " + name + "_*.png")

        lightness = X_test[0][i][:, :, 0]

        # Rescale L* back. [-1.0, 1.0] => [0.0, 100.0]
        lightness = (lightness + 1) * 50

        predicted = np.zeros((256, 256, 3))
        predicted[:, :, 0] = lightness
        predicted[:, :, 1:] = output[i]
        plt.imsave("result/predicted/" + name + ".jpeg", lab2rgb(predicted))

        if save_gray:
            bnw = np.zeros((256, 256, 3))
            bnw[:, :, 0] = lightness
            plt.imsave("result/bnw/" + name + ".jpeg", lab2rgb(bnw))

        if save_actual:
            actual = np.zeros((256, 256, 3))
            actual[:, :, 0] = lightness
            actual[:, :, 1:] = Y_test[i]
            plt.imsave("result/actual/" + name + ".jpeg", lab2rgb(actual))


if __name__ == '__main__':
    DATASET = '../data/imagenet/'
    training_files, testing_files = train_test_split(shuffle(os.listdir(DATASET)), test_size=0.15)
    model = encode_model()
    model.compile(optimizer=RMSprop(lr=1e-3), loss='mse')
    train(model, training_files, epochs=100)
    # test(model, testing_files, True, True)

    # filelist = shuffle(os.listdir('result/predicted/'))
    # filelist = filelist[:4]
    #
    # fig, ax = plt.subplots(4, 3, figsize=(16, 16))
    # row = 0
    # for filename in filelist:
    #     folder = 'result/bnw/'
    #     image_in = img_to_array(load_img(folder + filename))
    #     image_in = image_in / 255
    #     ax[row, 0].imshow(image_in)
    #
    #     folder = 'result/predicted/'
    #     image_in = img_to_array(load_img(folder + filename))
    #     image_in = image_in / 255
    #     ax[row, 1].imshow(image_in)
    #
    #     folder = 'result/actual/'
    #     image_in = img_to_array(load_img(folder + filename))
    #     image_in = image_in / 255
    #     ax[row, 2].imshow(image_in)
    #
    #     row += 1
