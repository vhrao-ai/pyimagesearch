# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense


# ------------------------
#   FireDetectionNetwork
# ------------------------
class FireDetectionNetwork:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be 'channels last' and the channels dimension itself
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1
        # CONV => RELU => POOL
        model.add(SeparableConv2D(16, (7, 7), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # CONV => RELU => POOL
        model.add(SeparableConv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # (CONV => RELU) * 2 => POOL
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Load first set of FC => RELU Layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Load the second set of FC => RELU Layers
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Softmax Classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # Return the constructed network architecture
        return model

