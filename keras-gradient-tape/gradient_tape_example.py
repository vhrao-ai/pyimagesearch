# -----------------------------
#   USAGE
# -----------------------------
# python gradient_tape_example.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
import time
import sys


# -----------------------------
#   FUNCTIONS
# -----------------------------
def build_model(width, height, depth, classes):
    # Initialize the input shape and channels dimension to be "channels last" ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # Build the model using Keras' Sequential API
    model = Sequential([
        # CONV => RELU => BN => POOL layer set
        Conv2D(16, (3, 3), padding="same", input_shape=inputShape),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        MaxPooling2D(pool_size=(2, 2)),
        # (CONV => RELU => BN) * 2 => POOL layer set
        Conv2D(32, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        Conv2D(32, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        MaxPooling2D(pool_size=(2, 2)),
        # (CONV => RELU => BN) * 3 => POOL layer set
        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        BatchNormalization(axis=chanDim),
        MaxPooling2D(pool_size=(2, 2)),
        # First (and only) set of FC => RELU layers
        Flatten(),
        Dense(256),
        Activation("relu"),
        BatchNormalization(),
        Dropout(0.5),
        # Softmax Classifier
        Dense(classes),
        Activation("softmax")
    ])
    # Return the built model to the calling function
    return model


def step(X, y):
    # Keep track of our gradients
    with tf.GradientTape() as tape:
        # Make a prediction using the model and then calculate the loss
        pred = model(X)
        loss = categorical_crossentropy(y, pred)
    # Calculate the gradients using our tape and then update the model weights
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))


# Initialize the number of epochs to train for, batch size, and the initial learning rate
EPOCHS = 25
BS = 64
INIT_LR = 1e-3

# Load the MNIST dataset
print("[INFO] Loading MNIST dataset...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# Add a channel dimension to every image in the dataset, then scale the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# One-hot encode the labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# Build the model and initialize the optimizer
print("[INFO] Creating model...")
model = build_model(28, 28, 1, 10)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# Compute the number of batch updates per epoch
numUpdates = int(trainX.shape[0] / BS)

# loop over the number of epochs
for epoch in range(0, EPOCHS):
    # show the current epoch number
    print("[INFO] Starting epoch {}/{}...".format(epoch + 1, EPOCHS), end="")
    sys.stdout.flush()
    epochStart = time.time()
    # Loop over the data in batch size increments
    for i in range(0, numUpdates):
        # Determine starting and ending slice indexes for the current batch
        start = i * BS
        end = start + BS
        # Take a step
        step(trainX[start:end], trainY[start:end])
    # Show timing information for the epoch
    epochEnd = time.time()
    elapsed = (epochEnd - epochStart) / 60.0
    print("Took {:.4} minutes".format(elapsed))

# In order to calculate accuracy using Keras' functions we first need to compile the model
model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=["acc"])

# Now that the model is compiled we can measure the accuracy
(loss, acc) = model.evaluate(testX, testY)
print("[INFO] Test accuracy: {:.4f}".format(acc))