import keras
from keras import Model
from keras.layers import Add, Activation, Conv2D, Dense, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D, MaxPool2D


class ResNetSettings:
    def __init__(self, unit, filter=(1, 1), strides=(1, 1), padding='valid'):
        self.unit = unit
        self.filter = filter
        self.strides = strides
        self.padding = padding


class ResNet:
    def __init__(self, input_tensor):
        x = self.conv1_layer(input_tensor)
        x = self.conv2_layer(x)
        x = self.conv3_layer(x)
        x = self.conv4_layer(x)
        x = self.conv5_layer(x)
        output_tensor = self.dense_layer(x)

        resnet50 = Model(input_tensor, output_tensor, name="ResNet")
        resnet50.summary()

    def resnet_layer(self, x, settings: list):
        x = Conv2D(settings[0].unit, settings[0].filter, settings[0].strides, settings[0].padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(settings[1].unit, settings[1].filter, settings[1].strides, settings[1].padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(settings[2].unit, settings[2].filter, settings[2].strides, settings[2].padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def conv1_layer(self, x):
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

        return x

    def conv2_layer(self, x):
        x = MaxPool2D((3, 3), 2)(x)
        skip = x
        settings = [
            ResNetSettings(64, (1, 1), (1, 1), 'valid'),
            ResNetSettings(64, (3, 3), (1, 1), 'same'),
            ResNetSettings(256, (1, 1), (1, 1), 'valid'),
        ]

        for i in range(3):
            if i == 0:
                # identity block
                x = self.resnet_layer(x, settings)
                skip = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(skip)
                skip = BatchNormalization()(skip)

                x = Add()([x, skip])
                x = Activation('relu')(x)
                skip = x
            else:
                x = self.resnet_layer(x, settings)
                x = Add()([x, skip])
                x = Activation('relu')(x)

        return x

    def conv3_layer(self, x):
        skip = x
        settings = [
            ResNetSettings(128, (1, 1), (2, 2), 'valid'),
            ResNetSettings(128, (3, 3), (1, 1), 'same'),
            ResNetSettings(512, (1, 1), (1, 1), 'valid'),
        ]

        for i in range(4):
            if i == 0:
                # identity block
                x = self.resnet_layer(x, settings)
                skip = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(skip)
                skip = BatchNormalization()(skip)

                x = Add()([x, skip])
                x = Activation('relu')(x)
                skip = x
                settings[0].strides = (1, 1)
            else:
                x = self.resnet_layer(x, settings)
                x = Add()([x, skip])
                x = Activation('relu')(x)

        return x

    def conv4_layer(self, x):
        skip = x
        settings = [
            ResNetSettings(256, (1, 1), (2, 2), 'valid'),
            ResNetSettings(256, (3, 3), (1, 1), 'same'),
            ResNetSettings(1024, (1, 1), (1, 1), 'valid'),
        ]

        for i in range(6):
            if i == 0:
                # identity block
                x = self.resnet_layer(x, settings)
                skip = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(skip)
                skip = BatchNormalization()(skip)

                x = Add()([x, skip])
                x = Activation('relu')(x)
                skip = x
                settings[0].strides = (1, 1)
            else:
                x = self.resnet_layer(x, settings)
                x = Add()([x, skip])
                x = Activation('relu')(x)

        return x

    def conv5_layer(self, x):
        skip = x
        settings = [
            ResNetSettings(512, (1, 1), (2, 2), 'valid'),
            ResNetSettings(512, (3, 3), (1, 1), 'same'),
            ResNetSettings(2048, (1, 1), (1, 1), 'valid'),
        ]

        for i in range(3):
            if i == 0:
                # identity block
                x = self.resnet_layer(x, settings)
                skip = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(skip)
                skip = BatchNormalization()(skip)

                x = Add()([x, skip])
                x = Activation('relu')(x)
                skip = x
                settings[0].strides = (1, 1)
            else:
                x = self.resnet_layer(x, settings)
                x = Add()([x, skip])
                x = Activation('relu')(x)

        return x

    def dense_layer(self, x):
        x = GlobalAveragePooling2D()(x)
        x = Dense(10, activation='softmax')(x)

        return x
