import keras
from vgg_config import *
from keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D, Dense
from keras.models import Sequential

class VGG(Sequential):
    def __init__(self, config:str):
        super().__init__(name=config)
        if config not in globals():
            print("%s config does not exist" % config)
        vgg = globals()[config]

        for i, conv_layer in enumerate(vgg):
            if i == len(vgg)-1:
                self.add(Flatten())
                for j in range(len(conv_layer)-1):
                    self.add(Dense(conv_layer[j], activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)))
                self.add(Dense(conv_layer[-1], activation='sigmoid'))
            else:
                for j in range(len(conv_layer)):
                    if j == 0 and i == 0:
                        self.add_conv(conv_layer[j], input_shape=(64, 64, 3))
                    else:
                        self.add_conv(conv_layer[j])

                    if j < len(conv_layer)-1:
                        self.add(Dropout(0.4))
                self.add(MaxPool2D(strides=(2, 2)))

    def add_conv(self, filters, input_shape=None):
        if input_shape is not None:
            self.add(Conv2D(filters, input_shape=input_shape,
                            kernel_size=kernel_size, padding=padding,
                            activation=activation,
                            kernel_regularizer=keras.regularizers.l2(l2_reg)))
        else:
            self.add(Conv2D(filters,
                            kernel_size=kernel_size, padding=padding,
                            activation=activation,
                            kernel_regularizer=keras.regularizers.l2(l2_reg)))
        self.add(BatchNormalization())

class VGG16(Sequential):
    def __init__(self):
        super().__init__(name="VGG16")
        self.conv1_layer()
        self.conv2_layer()
        self.conv3_layer()
        self.conv4_layer()
        self.conv5_layer()
        self.fc_layer()

    def conv1_layer(self):
        self.add(Conv2D(64, input_shape=(32,32,3), kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(MaxPool2D(strides=(2, 2)))

    def conv2_layer(self):
        self.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(MaxPool2D(strides=(2, 2)))

    def conv3_layer(self):
        self.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(MaxPool2D(strides=(2, 2)))

    def conv4_layer(self):
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(MaxPool2D(strides=(2, 2)))

    def conv5_layer(self):
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(BatchNormalization())
        self.add(MaxPool2D(strides=(2, 2)))

    def fc_layer(self):
        self.add(Flatten())
        self.add(Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)))
        self.add(Dense(10, activation='softmax'))

    if __init__ == "__main__":
        vgg = VGG('vgg11')
        vgg.summary()