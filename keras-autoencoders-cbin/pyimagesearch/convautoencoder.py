# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


# -----------------------------
#  Convolutional Autoencoder
# -----------------------------
class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latentDim=16):
        # Initialize the input shape to be "channels last" along with the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1
        # Define the input of the autoencoder
        inputs = Input(shape=inputShape)
        x = inputs
        # Loop over the number of filters
        for f in filters:
            # Apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
        # Flatten the network and then construct the latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim, name="encoded")(x)
        # Start building the decoder model which will accept the output of the encoder as its input
        x = Dense(np.prod(volumeSize[1:]))(latent)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        # Loop over the number of filters again, but this time in reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
        # Apply a single CONV_TRANSPOSE layer used to recover the original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid", name="decoded")(x)
        # Construct the autoencoder model
        autoencoder = Model(inputs, outputs, name="autoencoder")
        # Return the autoencoder model
        return autoencoder
