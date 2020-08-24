# -----------------------------
#   USAGE
# -----------------------------
# python train_ocr_model.py --az a_z_handwritten_data.csv --model handwriting.model

# -----------------------------
#   IMPORTS
# -----------------------------
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# Import the necessary packages
from pyimagesearch.models.resnet import ResNet
from pyimagesearch.az_dataset.helpers import load_mnist_dataset
from pyimagesearch.az_dataset.helpers import load_az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True, help="Path to A-Z dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="Path to output trained handwriting recognition model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Path to output training history file")
args = vars(ap.parse_args())

# Initialize the number of epochs to train for, initial learning rate and batch size
EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# Load the A-Z and MNIST datasets, respectively
print("[INFO] Loading datasets...")
(azData, azLabels) = load_az_dataset(args["az"])
(digitsData, digitsLabels) = load_mnist_dataset()

# The MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters
# are not incorrectly labeled as digits
azLabels += 10

# Stack the A-Z data and labels with the MNIST digits data and labels
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

# Each image in the A-Z and MNIST digts datasets are 28x28 pixels; however,
# the architecture we're using is designed for 32x32 images, so we need to resize them to 32x32
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# Add a channel dimension to every image in the dataset and scale the pixel intensities
# of the images from [0, 255] down to [0, 1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# Convert the labels from integers to vectors
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# Account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# Loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# Partition the data into training and testing splits using 80% of the data
# for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.05, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.15, horizontal_flip=False, fill_mode="nearest")

# Initialize and compile our deep neural network
print("[INFO] Compiling the model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] Training the network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, class_weight=classWeight, verbose=1)

# Define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# Evaluate the network
print("[INFO] Evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Save the model to disk
print("[INFO] Serializing network...")
model.save(args["model"], save_format="h5")

# Construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Initialize the list of output images
images = []

# Randomly select a few testing characters
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
    # Classify the character
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]
    # Extract the image from the test data and initialize the text label color as green (correct)
    image = (testX[i] * 255).astype("uint8")
    color = (0, 255, 0)
    # Otherwise, the class label prediction is incorrect
    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)
    # Merge the channels into one image, resize the image from 32x32 to 96x96 so we can better see it
    # and then draw the predicted label on the image
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    # Add the image to the list of output images
    images.append(image)

# Construct the montage for the images
montage = build_montages(images, (96, 96), (7, 7))[0]

# Show the output montage
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)