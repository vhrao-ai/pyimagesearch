# -----------------------------
#   USAGE
# -----------------------------
# python train.py --dataset dataset

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model", help="Path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# Grab the list of images in the dataset directory, then initialize the list of data (i.e, images) and class images
print("[INFO] Loading the images...")
img_paths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Loop over the image paths
for img_path in img_paths:
    # Extract the class label from the filename
    label = img_path.split(os.path.sep)[-2]
    # Load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # Update the data and labels list respectively
    data.append(img)
    labels.append(label)

# Convert the data and labels to Numpy arrays while scaling the pixel intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

# Perform the one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Partition the data into training and testing splits using 80% of the data for training
# and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Initialize the training data augmentation object
train_aug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# Load the VGG16 network, ensuring the head FC layer sets are left off
base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(64, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# Place the head FC model on top of the base model (this will become the actual model that will be trained)
model = Model(inputs=base_model.input, outputs=head_model)

# Loop over the all layers in the base model and freeze them so they will *not* be updated during the first training
for layer in base_model.layers:
    layer.trainable = False

# Print compile the model
print("[INFO] Compiling the model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the head of the network
print("[INFO] Training the head of the network...")
H = model.fit_generator(train_aug.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS,
                        validation_data=(testX, testY), validation_steps=len(testX) // BS, epochs=EPOCHS)

# Make predictions on the testing set
print("[INFO] Evaluating the network...")
pred_idxs = model.predict(testX, batch_size=BS)

# For each image in the testing set we need to find the index of the label
# with corresponding largest predicted probability
pred_idxs = np.argmax(pred_idxs, axis=1)

# Show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), pred_idxs, target_names=lb.classes_))

# Compute the confusion matrix and use it to derive the raw accuracy, sensitivity and specificity
cm = confusion_matrix(testY.argmax(axis=1), pred_idxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# Show the confusion matrix, accuracy, sensitivity and specificity
print(cm)
print("ACC: {:.4f}".format(acc))
print("Sensitivity: {:.4f}".format(sensitivity))
print("Specificity: {:.4f}".format(specificity))

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Serialize the model to disk
print("[INFO] Saving the COVID-19 detector model...")
model.save(args["model"], save_format="h5")



