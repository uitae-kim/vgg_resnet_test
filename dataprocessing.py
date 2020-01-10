import numpy as np
import random
import keras
import matplotlib.pyplot as plt
import cv2

def patch(x_index: list, x: np.ndarray):
    shape = x[x_index[0]].shape
    x_left = [0, 0, shape[0], shape[0]]
    y_top = [0, shape[0], 0, shape[0]]
    reference_shape = (shape[0]*2, shape[1]*2, shape[2])

    result = np.zeros(reference_shape)

    for i in range(4):
        offsets = (x_left[i], y_top[i], 0)
        insert_here = [slice(offsets[dim],
                       offsets[dim]+x[x_index[i]].shape[dim]) for dim in range(x[x_index[i]].ndim)]
        insert_here = tuple(insert_here)
        result[insert_here] = x[x_index[i]]

    return result


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

    label_one = []
    label_zero = []

    # 2 ~ 7이 동물이 들어가있음 --> 0
    for i, y in enumerate(y_train):
        if 2 <= y <= 7:
            y_train[i] = 0
            label_zero.append(i)
        else:
            y_train[i] = 1
            label_one.append(i)

    x_train_new = np.zeros((x_train.shape[0], ref[0], ref[1], ref[2]), dtype=x_train[0].dtype)
    """
    for i, x in enumerate(x_train):
    x_train_new[i] = pad(x)
    """

    x_index = []

    for i in range(50000):
        if i < 20000:
            zero_cnt = 4
        else:
            zero_cnt = np.random.randint(1, 3)
        rand_num = np.random.randint(0, len(label_zero)-1)

        for j in range(zero_cnt):
            while rand_num in x_index:
                rand_num = np.random.randint(0, len(label_zero) - 1)
            x_index.append(rand_num)

        rand_num = np.random.randint(0, len(label_one)-1)
        for j in range(4-zero_cnt):
            while rand_num in x_index:
                rand_num = np.random.randint(0, len(label_one) - 1)
            x_index.append(rand_num)

        random.shuffle(x_index)

        x_train_new[i] = patch(x_index, x_train)
        x_index.clear()

    label_one.clear()
    label_zero.clear()

    for i, y in enumerate(y_test):
        if 2 <= y <= 7:
            y_test[i] = 0
            label_zero.append(i)
        else:
            y_test[i] = 1
            label_one.append(i)

    x_test_new = np.zeros((x_test.shape[0], ref[0], ref[1], ref[2]), dtype=x_test[0].dtype)
    """
    for i, x in enumerate(x_test):
    x_test_new[i] = pad(x)
    """

    for i in range(10000):
        if i < 40000:
            zero_cnt = 4
        else:
            zero_cnt = np.random.randint(1, 3)
        rand_num = np.random.randint(0, len(label_zero)-1)

        for j in range(zero_cnt):
            while rand_num in x_index:
                rand_num = np.random.randint(0, len(label_zero) - 1)
            x_index.append(rand_num)

        rand_num = np.random.randint(0, len(label_one)-1)
        for j in range(4-zero_cnt):
            while rand_num in x_index:
                rand_num = np.random.randint(0, len(label_one) - 1)
            x_index.append(rand_num)

        random.shuffle(x_index)

        x_test_new[i] = patch(x_index, x_test)
        x_index.clear()

    return (x_train_new, y_train), (x_test_new, y_test)

