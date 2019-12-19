import keras
from vgg import VGG16
import sys


def main(argv):
    dataset = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    model = VGG16()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, batch_size=50, epochs=int(argv[1])) #0번 index에 main.py가 들어감


if __name__=="__main__":
    main(sys.argv)
