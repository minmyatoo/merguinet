from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
)
from tensorflow.keras import backend as K


class MerguiNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        def add_conv_block(filters, kernel_size, padding="same"):
            model.add(Conv2D(filters, kernel_size, padding=padding))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(Conv2D(filters, kernel_size, padding=padding))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

        # First Convolutional Block
        add_conv_block(32, (3, 3))

        # Second Convolutional Block
        add_conv_block(64, (3, 3))

        # Third Convolutional Block
        add_conv_block(128, (3, 3))
        add_conv_block(64, (3, 3))

        # Fourth Convolutional Block
        add_conv_block(128, (3, 3))
        add_conv_block(128, (3, 3))

        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Model Summary
        model.summary()

        return model
