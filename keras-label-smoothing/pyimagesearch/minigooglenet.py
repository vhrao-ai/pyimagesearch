# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# -----------------------------
#   MINI GOOGLE NETWORK
# -----------------------------
class MiniGoogleNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chan_dim, padding="same"):
        # Define a CONV => BN => RELU pattern
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chan_dim)
        x = Activation("relu")(x)
        # Return the block
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chan_dim):
        # Define two CONV modules, then concatenate across the channel dimension
        conv_1x1 = MiniGoogleNet.conv_module(x, numK1x1, 1, 1, (1, 1), chan_dim)
        conv_3x3 = MiniGoogleNet.conv_module(x, numK3x3, 3, 3, (1, 1), chan_dim)
        x = concatenate([conv_1x1, conv_3x3], axis=chan_dim)
        # Return the block
        return x

    @staticmethod
    def downsample_module(x, K, chan_dim):
        # Define the CONV module and POOL, then concatenate across the channel dimensions
        conv_3x3 = MiniGoogleNet.conv_module(x, K, 3, 3, (2, 2), chan_dim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chan_dim)
        # Return the block
        return x

    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the input shape to be "channels last" and the channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1
        # If we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        # Define the model input and first CONV module
        inputs = Input(shape=input_shape)
        x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
        # Two Inception modules followed by a downsample module
        x = MiniGoogleNet.inception_module(x, 32, 32, chan_dim)
        x = MiniGoogleNet.inception_module(x, 32, 48, chan_dim)
        x = MiniGoogleNet.downsample_module(x, 80, chan_dim)
        # Four Inception modules followed by a downsample module
        x = MiniGoogleNet.inception_module(x, 112, 48, chan_dim)
        x = MiniGoogleNet.inception_module(x, 96, 64, chan_dim)
        x = MiniGoogleNet.inception_module(x, 80, 80, chan_dim)
        x = MiniGoogleNet.inception_module(x, 48, 96, chan_dim)
        x = MiniGoogleNet.downsample_module(x, 96, chan_dim)
        # Two Inception modules followed by global POOL and dropout
        x = MiniGoogleNet.inception_module(x, 176, 160, chan_dim)
        x = MiniGoogleNet.inception_module(x, 176, 160, chan_dim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)
        # Softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)
        # Create the model
        model = Model(inputs, x, name="googlenet")
        # Return the constructed network architecture
        return model
