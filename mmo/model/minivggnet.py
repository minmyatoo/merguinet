from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Flatten,
    Dropout,
    Dense,
    Input,
)

class MerguiNet:

    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)

        # Load the pre-trained VGG16 model without the top classification layers
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=inputShape))

        # Freeze the weights of the pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output

        # Add custom layers on top of the pre-trained model
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)

        # Model Summary
        model.summary()

        return model
