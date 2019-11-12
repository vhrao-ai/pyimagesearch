# ------------------------
#   USAGE
# ------------------------
# python train.py --lr-find 1
# python train.py

# ------------------------
#   IMPORTS
# ------------------------
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# Import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .pyimagesearch.learningratefinder import LearningRateFinder
from .pyimagesearch.clr_callback import CyclicLearningRate
from .pyimagesearch import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import sys
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0, help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())
# Grab the paths to all the image in the dataset directory and initialize the list of images and class labels
print("[INFO] Loading the images...")
imgPaths = list(paths.list_images(config.DATASET_PATH))
data = []
labels = []
# Loop over the image paths
for imgPath in imgPaths:
    # extract the class label
    label = imgPath.split(os.path.sep)[-2]
    # load the image, convert it to RGB channel ordering
    # and resize the image to a fixed size 224x224 (ignore aspect ratio)
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # update the data and labels lists respectively
    data.append(img)
    labels.append(label)
# Convert the data and labels to NumPy arrays
print("[INFO] Processing the data...")
data = np.array(data, dtype="float32")
labels = np.array(labels)
# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# Partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=config.TEST_SPLIT, random_state=42)
# Take the validation split from the training split
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=config.VAL_SPLIT, random_state=84)
# Initialize the training data augmentation object
aug = ImageDataGenerator(rotation_range=30, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
# Load the VGG16 network, ensuring the head FC layers sets are left off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)
# Place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# Loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False
# Compile the model (this needs to be done after our setting our layers to being non-trainable)
print("[INFO] Compiling the model...")
opt = SGD(lr=config.MIN_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# Check to see if we are attempting to find an optimal learning rate before training for the full number of epochs
if args["lr_find"] > 0:
    # initialize the learning rate finder and then train with learning rates ranging from 1e-10 to 1e+1
    print("[INFO] Finding the learning rate...")
    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), 1e-10, 1e+1,
             steps_per_epoch=np.ceil((trainX.shape[0] / float(config.BATCH_SIZE))),
             epochs=20,
             batch_size=config.BATCH_SIZE)
    # plot the loss for the various learning rates and save the resulting plot to disk
    lrf.plot_loss()
    plt.savefig(config.LRFIND_PLOT_PATH)
    # gracefully exit the script so we can adjust our learning rates in the config
    # and then train the network for our full set of epochs
    print("[INFO] Learning rate finder complete")
    print("[INFO] Examine plot and adjust learning rates before training")
    sys.exit(0)
# Otherwise, we have already defined a learning rate space to train over, so compute the step size
# and initialize the cyclic learning rate method
stepSize = config.STEP_SIZE * (trainX.shape[0] // config.BATCH_SIZE)
clr = CyclicLearningRate(mode=config.CLR_METHOD, base_lr=config.MIN_LR, max_lr=config.MAX_LR, step_size=stepSize)
# Train the network
print("[INFO] Training the network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), validation_data=(valX, valY),
                        steps_per_epoch=trainX.shape[0] // config.BATCH_SIZE, epochs=config.NUM_EPOCHS,
                        callbacks=[clr], verbose=1)
# Evaluate the network and show a classification report
print("[INFO] Evaluating the network...")
predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=config.CLASSES))
# Serialize the model to disk
print("[INFO] Serializing the network to '{}'...".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)
# Construct a plot that plots and saves the training history
N = np.arange(0, config.NUM_EPOCHS)
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
plt.savefig(config.TRAINING_PLOT_PATH)
# Plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)