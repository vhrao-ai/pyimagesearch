# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
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


# ------------------------
#   CONVAUTOENCODER CLASS
# ------------------------
class ConvAutoEncoder:
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latent_dim=16):
        # Initialize the input shape to be "channels last" along with the channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1
        # Define the input to the encoder
        inputs = Input(shape=input_shape)
        x = inputs
        # Loop over the number of filters
        for f in filters:
            # Apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chan_dim)(x)
        # Flatten the network and then construct the latent vector
        volume_size = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latent_dim)(x)
        # Build the encoder model
        encoder = Model(inputs, latent, name="encoder")
        # Start the building to the decoder model which will accept the output of the encoder as its inputs
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(np.prod(volume_size[1:]))(latent_inputs)
        x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
        # Loop over the number of filters again, but this time in reverse order
        for f in filters[::-1]:
            # Apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chan_dim)(x)
        # Apply a single CONV_TRANSPOSE layer used to recover the original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        # Build the decoder model
        decoder = Model(latent_inputs, outputs, name="decoder")
        # The autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
        # Return a 3-tuple of the encoder, decoder and autoencoder
        return encoder, decoder, autoencoder

