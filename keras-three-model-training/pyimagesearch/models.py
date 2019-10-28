# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from keras.models import Model
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate


# ------------------------
#   FUNCTIONS
# ------------------------
def shallownet_sequential(width, height, depth, classes):
    # Initialize the model along with the input shape to be "channels last" ordering
    model = Sequential()
    input_shape = (height, width, depth)
    # Define the first (and only) CONV => RELU layer
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    # Softmax classifier
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # Return the constructed network architecture
    return model


def minigooglenet_functional(width, height, depth, classes):
    def conv_module(x, K, kX, kY, stride, chan_dim, padding="same"):
        # define a CONV => BN => RELU pattern
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation("relu")(x)
        # return the block
        return x

    def inception_module(x, num_k1x1, num_k3x3, chan_dim):
        # define two CONV modules, then concatenate across the channel dimension
        conv_1x1 = conv_module(x, num_k1x1, 1, 1, (1, 1), chan_dim)
        conv_3x3 = conv_module(x, num_k3x3, 3, 3, (1, 1), chan_dim)
        x = concatenate([conv_1x1, conv_3x3], axis=chan_dim)
        # return the block
        return x

    def downsample_module(x, K, chan_dim):
        # define the CONV module and POOL, then concatenate across the channel dimensions
        conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chan_dim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chan_dim)
        # return the block
        return x

    # Initialize the input shape to be "channels last" and the channels dimension itself
    input_shape = (height, width, depth)
    chan_dim = -1
    # Define the model input and first CONV module
    inputs = Input(shape=input_shape)
    x = conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)
    # Create two inception module followed by a downsample module
    x = inception_module(x, 32, 32, chan_dim)
    x = inception_module(x, 32, 48, chan_dim)
    x = downsample_module(x, 80, chan_dim)
    # Create four inception modules followed by a downsample module
    x = inception_module(x, 112, 48, chan_dim)
    x = inception_module(x, 96, 64, chan_dim)
    x = inception_module(x, 80, 80, chan_dim)
    x = inception_module(x, 48, 96, chan_dim)
    x = downsample_module(x, 96, chan_dim)
    # Create two inception modules followed by global POOL and dropout
    x = inception_module(x, 176, 160, chan_dim)
    x = inception_module(x, 176, 160, chan_dim)
    x = AveragePooling2D((7, 7))(x)
    x = Dropout(0.5)(x)
    # Softmax Classifier
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)
    # Create the model
    model = Model(inputs, x, name="minigooglenet")
    # Return the constructed network architecture
    return model


# ------------------------
#  MiniVGGNetModel Class
# ------------------------
class MiniVGGNetModel(Model):
    def __init__(self, classes, chan_dim=1):
        # call the parent constructor
        super(MiniVGGNetModel, self).__init__()
        # initialize the layers in the first (CONV => RELU) * 2 => POOL layer set
        self.conv1A = Conv2D(32, (3, 3), padding="same")
        self.act1A = Activation("relu")
        self.bn1A = BatchNormalization(axis=chan_dim)
        self.conv1B = Conv2D(32, (3, 3), padding="same")
        self.act1B = Activation("relu")
        self.bn1B = BatchNormalization(axis=chan_dim)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        # initialize the layers in the second (CONV => RELU) * 2 => POOL layer set
        self.conv2A = Conv2D(32, (3, 3), padding="same")
        self.act2A = Activation("relu")
        self.bn2A = BatchNormalization(axis=chan_dim)
        self.conv2B = Conv2D(32, (3, 3), padding="same")
        self.act2B = Activation("relu")
        self.bn2B = BatchNormalization(axis=chan_dim)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        # initialize the layers in our fully-connected layer set
        self.flatten = Flatten()
        self.dense3 = Dense(512)
        self.act3 = Activation("relu")
        self.bn3 = BatchNormalization()
        self.do3 = Dropout(0.5)
        # initialize the layers in the softmax classifier layer set
        self.dense4 = Dense(classes)
        self.softmax = Activation("softmax")

    def call(self, inputs, mask=None):
        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(inputs)
        x = self.act1A(x)
        x = self.bn1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.bn1B(x)
        x = self.pool1(x)
        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv2A(inputs)
        x = self.act2A(x)
        x = self.bn2A(x)
        x = self.conv2B(x)
        x = self.act2B(x)
        x = self.bn2B(x)
        x = self.pool2(x)
        # build our FC layer set
        x = self.flatten(x)
        x = self.dense3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.do3(x)
        # build the softmax classifier
        x = self.dense4(x)
        x = self.softmax(x)
        # return the constructed model
        return x