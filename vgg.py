import keras
from keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D, Dense
from keras.models import Sequential

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