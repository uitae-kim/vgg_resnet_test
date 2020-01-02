import numpy as np
import random
import keras
import matplotlib.pyplot as plt
import cv2

def pad(x):
    shape = x.shape
    reference_shape = (shape[0] * 2, shape[1] * 2, shape[2])
    x = randomize(x)
    x_left = random.randint(0, reference_shape[0] - x.shape[0])
    y_top = random.randint(0, reference_shape[1] - x.shape[1])
    offsets = (x_left, y_top, 0)

    result = np.random.randint(0, 255, reference_shape)
    insert_here = [slice(offsets[dim], offsets[dim]+x.shape[dim]) for dim in range(x.ndim)]
    insert_here = tuple(insert_here)
    result[insert_here] = x
    return result

def randomize(x: np.ndarray):
    shape = x.shape
    new_shape = (int(shape[0] * (random.random() + 0.5)),
                 int(shape[1] * (random.random() + 0.5)))
    x = cv2.resize(x, dsize=new_shape, interpolation=cv2.INTER_CUBIC)

    return x

def preprocess():
    dataset = keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    shape = x_train[0].shape
    ref = (shape[0] * 2, shape[1] * 2, shape[2])

    x_train_new = np.zeros((x_train.shape[0], ref[0], ref[1], ref[2]), dtype=x_train[0].dtype)
    for i, x in enumerate(x_train):
        x_train_new[i] = pad(x)

    x_test_new = np.zeros((x_test.shape[0], ref[0], ref[1], ref[2]), dtype=x_test[0].dtype)
    for i, x in enumerate(x_test):
        x_test_new[i] = pad(x)

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

    return (x_train_new, y_train), (x_test_new, y_test)