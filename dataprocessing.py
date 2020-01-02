import keras
import matplotlib.pyplot as plt

def preprocess():
    dataset = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train, x_test = normalize(x_train.astype('float32'), x_test.astype('float32'))
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # 2 ~ 7이 동물이 들어가있음 --> 0
    for i, y in enumerate(y_train):
        if 2 <= y <= 7:
            y_train[i] = 0
        else:
            y_train[i] = 1

    for i, y in enumerate(y_test):
        if 2 <= y <= 7:
            y_test[i] = 0
        else:
            y_test[i] = 1

    return (x_train, y_train), (x_test, y_test)