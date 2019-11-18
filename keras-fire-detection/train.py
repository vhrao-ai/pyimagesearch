# ----------------------
#   USAGE
# ----------------------
# python train.py --lr-find 1
# python train.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Set the matplotlib backed so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.learning_rate_finder import LearningRateFinder
from pyimagesearch.fire_detection_network import FireDetectionNetwork
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys
import config_params


# -----------------------------
#   Load Dataset Function
# -----------------------------
def load_dataset(dataset_path):
    # Grab the paths to all images in our dataset directory, then initialize our lists of images
    image_paths = list(paths.list_images(dataset_path))
    data = []
    # Loop over the image paths
    for imagePath in image_paths:
        # Load the image and resize it to be a fixed 128x128 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (128, 128))
        # Add the image to the data lists
        data.append(image)
    # Return the data list as a NumPy array
    return np.array(data, dtype="float32")


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0, help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

# Load the fire and non-fire images
print("[INFO] Loading training dataset...")
fireData = load_dataset(config_params.FIRE_PATH)
nonFireData = load_dataset(config_params.NON_FIRE_PATH)

# Construct the class labels for the data
fireLabels = np.ones((fireData.shape[0],))
nonFireLabels = np.zeros((nonFireData.shape[0],))

# Stack the fire data with the non-fire data, then scale the data to the range [0, 1]
data = np.vstack([fireData, nonFireData])
labels = np.hstack([fireLabels, nonFireLabels])
data /= 255

# Perform one-hot encoding on the labels and account for skew in the labeled data
labels = to_categorical(labels, num_classes=2)
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# Construct the training and testing split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=config_params.TEST_SPLIT, random_state=42)

# Initialize the training data augmentation object
aug = ImageDataGenerator(rotation_range=30, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Initialize the optimizer and model
print("[INFO] Compiling the network model...")
opt = SGD(lr=config_params.INIT_LR, momentum=0.9, decay=config_params.INIT_LR / config_params.NUM_EPOCHS)
model = FireDetectionNetwork.build(width=128, height=128, depth=3, classes=2)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Check to see if we are attempting to find an optimal learning rate before training for the full number of epochs
if args["lr_find"] > 0:
    # initialize the learning rate finder and then train with learning rates ranging from 1e-10 to 1e+1
    print("[INFO] Finding the learning rate...")
    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(trainX, trainY, batch_size=config_params.BATCH_SIZE), 1e-10, 1e+1,
             steps_per_epoch=np.ceil((trainX.shape[0] / float(config_params.BATCH_SIZE))),
             epochs=20, batch_size=config_params.BATCH_SIZE, class_weight=classWeight)
    # plot the loss for the various learning rates and save the resulting plot to disk
    lrf.plot_loss()
    plt.savefig(config_params.LRFIND_PLOT_PATH)
    # gracefully exit the script so we can adjust our learning rates in the config and then train the network
    # for our full set of epochs
    print("[INFO] Learning rate finder complete")
    print("[INFO] Examine plot and adjust learning rates before training")
    sys.exit(0)

# Train the network
print("[INFO] Training the network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=config_params.BATCH_SIZE), validation_data=(testX, testY),
                        steps_per_epoch=trainX.shape[0] // config_params.BATCH_SIZE, epochs=config_params.NUM_EPOCHS,
                        class_weight=classWeight, verbose=1)

# Evaluate the network and show a classification report
print("[INFO] Evaluating the network...")
predictions = model.predict(testX, batch_size=config_params.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=config_params.CLASSES))

# Serialize the model to disk
print("[INFO] Serializing the network to '{}'...".format(config_params.MODEL_PATH))
model.save(config_params.MODEL_PATH)

# Construct a plot that plots and saves the training history
N = np.arange(0, config_params.NUM_EPOCHS)
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
plt.savefig(config_params.TRAINING_PLOT_PATH)
