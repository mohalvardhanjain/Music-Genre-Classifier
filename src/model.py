from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(
    input_shape=(150, 150, 1),
    num_classes=10
):

    model = keras.Sequential([

        layers.Input(
            shape=input_shape
        ),

        # Block 1
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 2
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 3
        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(
            0.3
        ),

        # Block 4
        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.Conv2D(
            256,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 5
        layers.Conv2D(
            512,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.Conv2D(
            512,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(
            0.4
        ),

        # Classifier
        layers.Flatten(),

        layers.Dense(
            1200,
            activation="relu"
        ),

        layers.Dropout(
            0.5
        ),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model