import tensorflow as tf
import keras
from keras import Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from vgg import VGG16, VGG
from resnet import ResNet
import numpy as np
import sys
import dataprocessing

def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    return X_train, X_test


def main_vgg(argv):
    # dataset = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataprocessing.preprocess() # dataset.load_data()

    x_train, x_test = normalize(x_train.astype('float32'), x_test.astype('float32'))
    # y_train = keras.utils.to_categorical(y_train, num_classes=10)
    # y_test = keras.utils.to_categorical(y_test, num_classes=10)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )

    datagen.fit(x_train)

    model = VGG(argv[1])

    lr = 0.1
    lr_decay = 1e-6
    lr_drop = 20

    def lr_scheduler(epoch):
        return lr * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    sgd = SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)

    model_cp = keras.callbacks.ModelCheckpoint("Model/{0}.model".format(argv[1]))

    model.compile(optimizer=sgd, loss='crossentropy', metrics=['accuracy'])
    print(model.summary())
    batch_size = 128
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=int(argv[2]),
                        validation_data=(x_test, y_test),
                        callbacks=[reduce_lr, model_cp])
    print(model.evaluate(x_test, y_test))

    model.save("Model/{0}.model".format(argv[1]))
    model.save_weights("Model/{0}.weights".format(argv[1]))


def main_resnet(argv):
    dataset = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    resnet = ResNet(Input(shape=(32, 32, 3), dtype='float32', name='input'))
    resnet.model.fit(x_train, y_train, batch_size=50, epochs=int(argv[2]))


if __name__=="__main__":
    if "vgg" in sys.argv[1]:
        main_vgg(sys.argv)
    elif sys.argv[1] == "resnet":
        main_resnet(sys.argv)
