# -----------------------------
#   USAGE
# -----------------------------
# python train.py --dataset generated_dataset --plot plot_generated_dataset.png
# python train.py --dataset dogs_vs_cats_small --plot plot_dogs_vs_cats_no_aug.png
# python train.py --dataset dogs_vs_cats_small --augment 1 --plot plot_dogs_vs_cats_with_aug.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# Import the necessary packages
from pyimagesearch.resnet import ResNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-a", "--augment", type=int, default=-1,
                help="whether or not 'on the fly' data augmentation should be used")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Initialize the initial learning rate, batch size, and number of epochs to train for
INIT_LR = 1e-1
BS = 8
EPOCHS = 50

# Grab the list of images in the dataset directory, then initialize the list of data (i.e, images) and class images
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Loop over the image paths
for imagePath in imagePaths:
    # Extract the class label from the filename, load the image and resize it to be a fixed 64x64 pixels,
    # ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    # Update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# Convert the data into a NumPy array, then preprocess it by scaling all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# Encode the labels (which are currently strings) as integers and then one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# Partition the data into training and testing splits using 75% for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Initialize the data augmenter as an "empty" image data generator
aug = ImageDataGenerator()

# Check to see if we are applying "on the fly" data augmentation and if so re-instantiate the object
if args["augment"] > 0:
    print("[INFO] Performing 'on the fly' data augmentation")
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Initialize the optimizer and model
print("[INFO] Compiling the model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)
model = ResNet.build(64, 64, 3, 2, (2, 3, 4), (32, 64, 128, 256), reg=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] Training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)

# Evaluate the network
print("[INFO] Evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# Plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_accuracy")
plt.plot(N, H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])