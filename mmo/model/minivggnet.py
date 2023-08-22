from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
    Input,
)
from tensorflow.keras import backend as K

class MerguiNet:

    @staticmethod
    def _conv_block(x, filters, kernel_size, padding="same"):
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)  # Use default axis (-1 for channels_last)
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        return x

    @staticmethod
    def _fully_connected_block(x, units):
        x = Dense(units)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        return x

    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1 if K.image_data_format() == "channels_last" else 1

        input_layer = Input(shape=inputShape)
        x = input_layer

        # Convolutional Blocks
        x = MerguiNet._conv_block(x, 32, (3, 3))
        x = MerguiNet._conv_block(x, 64, (3, 3))
        x = MerguiNet._conv_block(x, 128, (3, 3))
        x = MerguiNet._conv_block(x, 64, (3, 3))
        x = MerguiNet._conv_block(x, 128, (3, 3))
        x = MerguiNet._conv_block(x, 128, (3, 3))

        # Fully Connected Layers
        x = Flatten()(x)
        x = MerguiNet._fully_connected_block(x, 512)

        # Output Layer
        output_layer = Dense(classes, activation="softmax")(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        # Model Summary
        model.summary()

        return model
