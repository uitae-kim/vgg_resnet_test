import os
import numpy as np
import random
import keras
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import load_img
import PIL

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

def patch_internal():
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
    y_train_new = np.zeros(y_train.shape, dtype=y_train[0].dtype)

    """
    for i, x in enumerate(x_train):
    x_train_new[i] = pad(x)
    """

    for i in range(50000):
        data_index = []
        if i < 20000:
            zero_cnt = 4
        else:
            zero_cnt = np.random.randint(1, 3)
        rand_num = np.random.randint(0, len(label_zero) - 1)

        for j in range(zero_cnt):
            while label_zero[rand_num] in data_index:
                rand_num = np.random.randint(0, len(label_zero) - 1)
            data_index.append(label_zero[rand_num])

        rand_num = np.random.randint(0, len(label_one) - 1)
        for j in range(4 - zero_cnt):
            while label_one[rand_num] in data_index:
                rand_num = np.random.randint(0, len(label_one) - 1)
            data_index.append(label_one[rand_num])

        random.shuffle(data_index)

        x_train_new[i] = patch(data_index, x_train)
        # zero count가 0 레이블의 개수이므로, 이렇게 연산하면 zero가 하나도 없는 경우에만 0이 됨
        y_train_new[i] = 1 - zero_cnt // 4

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
    y_test_new = np.zeros(y_test.shape, dtype=y_test[0].dtype)
    """
    for i, x in enumerate(x_test):
    x_test_new[i] = pad(x)
    """

    for i in range(10000):
        data_index = []
        if i < 4000:
            zero_cnt = 4
        else:
            zero_cnt = np.random.randint(1, 3)
        rand_num = np.random.randint(0, len(label_zero) - 1)

        for j in range(zero_cnt):
            while label_zero[rand_num] in data_index:
                rand_num = np.random.randint(0, len(label_zero) - 1)
            data_index.append(label_zero[rand_num])

        rand_num = np.random.randint(0, len(label_one) - 1)
        for j in range(4 - zero_cnt):
            while label_one[rand_num] in data_index:
                rand_num = np.random.randint(0, len(label_one) - 1)
            data_index.append(label_one[rand_num])

        random.shuffle(data_index)

        x_test_new[i] = patch(data_index, x_test)
        # zero count가 0 레이블의 개수이므로, 이렇게 연산하면 zero가 하나도 없는 경우에만 0이 됨
        y_test_new[i] = 1 - zero_cnt // 4

    return (x_train_new, y_train_new), (x_test_new, y_test_new)

def preprocess(type="bg"):
    if type=="patch":
        (x_train_new, y_train_new), (x_test_new, y_test_new) = patch_internal()
        return (x_train_new, y_train_new), (x_test_new, y_test_new)
    else:
        background = []
        path = "/Users/DavidJeong 1/Desktop"
        l = os.listdir(path)
        for p in l:
            if "Road" in p:
                img = cv2.imread(path+"/"+p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                background.append(img)

        dataset = keras.datasets.cifar10

        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        ref = (128, 128, 3)

        # 2 ~ 7이 동물이 들어가있음 --> 0
        for i, y in enumerate(y_train):
            if 2 <= y <= 7:
                y_train[i] = 0
            else:
                y_train[i] = 1

        x_train_new = np.zeros((x_train.shape[0], ref[0], ref[1], ref[2]), dtype=x_train[0].dtype)
        y_train_new = np.zeros(y_train.shape, dtype=y_train[0].dtype)

        x_test_new = np.zeros((x_test.shape[0], ref[0], ref[1], ref[2]), dtype=x_test[0].dtype)
        y_test_new = np.zeros(y_test.shape, dtype=y_test[0].dtype)

        for i in range(50000):
            rand_num = np.random.randint(0, 3)
            x_train_new[i] = background[rand_num]

            x = x_train[i]

            reference_shape = (128, 128, 3)
            x_left = random.randint(0, reference_shape[0] - x.shape[0])
            y_top = random.randint(0, reference_shape[1] - x.shape[1])
            offsets = (x_left, y_top, 0)

            insert_here = [slice(offsets[dim], offsets[dim] + x.shape[dim]) for dim in range(x.ndim)]
            insert_here = tuple(insert_here)
            x_train_new[i][insert_here] = x

        for i in range(10000):
            rand_num = np.random.randint(0, 3)
            x_test_new[i] = background[rand_num]

            x = x_test[i]

            reference_shape = (128, 128, 3)
            x_left = random.randint(0, reference_shape[0] - x.shape[0])
            y_top = random.randint(0, reference_shape[1] - x.shape[1])
            offsets = (x_left, y_top, 0)

            insert_here = [slice(offsets[dim], offsets[dim] + x.shape[dim]) for dim in range(x.ndim)]
            insert_here = tuple(insert_here)
            x_test_new[i][insert_here] = x

        return (x_train_new, y_train_new), (x_test_new, y_test_new)











