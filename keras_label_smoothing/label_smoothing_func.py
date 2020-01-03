# -----------------------------
#   USAGE
# -----------------------------
# python label_smoothing_func.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# Import the necessary packages
from keras_label_smoothing.pyimagesearch.learning_rate_schedulers import PolynomialDecay
from keras_label_smoothing.pyimagesearch.minigooglenet import MiniGoogleNet
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


# -----------------------------
#   FUNCTIONS
# -----------------------------
def smooth_labels(labels, factor=0.1):
    # Smooth the labels
    labels += (1 - factor)
    labels += (factor / labels.shape[1])
    # Return the smoothed labels
    return labels


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--smoothing", type=float, default=0.1, help="amount of label smoothing to be applied")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output plot file")
args = vars(ap.parse_args())

# Define the total number of epochs to train for initial learning rate and batch size
NUM_EPOCHS = 70
INIT_LR = 5e-3
BATCH_SIZE = 64

# Initiliaze the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load the training and testing data, converting the images from integers to floats
print("[INFO] Loading CIFAR-10 dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# Apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# Convert the labels from integers to vectors, converting the darta type to floats so we can apply label smoothing
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
trainY = trainY.astype("float")
testY = testY.astype("float")

# Apply label smoothing to the training labels
print("[INFO] Smoothing amount: {}".format(args["smoothing"]))
print("[INFO] Before smoothing: {}".format(trainY[0]))
trainY = smooth_labels(trainY, args["smoothing"])
print("[INFO] After smoothing: {}".format(trainY[0]))

# Construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

# Construct the learning rate scheduler callback
schedule = PolynomialDecay(max_epochs=NUM_EPOCHS, init_alpha=INIT_LR, power=1.0)
callbacks = [LearningRateScheduler(schedule)]

# Initialize the optimizer and the model
print("[INFO] Compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogleNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] Training the network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

# Evaluate the network
print("[INFO] Evaluating the network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Construct the plot that plots and saves the training history
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
