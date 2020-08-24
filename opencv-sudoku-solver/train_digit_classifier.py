# -----------------------------
#   USAGE
# -----------------------------
# python train_digit_classifier.py --model output/digit_classifier.h5

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.models.sudokunet import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to output model after training")
args = vars(ap.parse_args())

# Initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 128

# Grab the MNIST dataset
print("[INFO] Accessing MNIST dataset...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# Add a channel (i.e, grayscale) dimension to the digits
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# Scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# Convert the labels from the integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# Initialize the optimizer and model
print("[INFO] Compiling the model...")
opt = Adam(lr=INIT_LR)
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] Training the network...")
H = model.fit(trainData, trainLabels, validation_data=(testData, testLabels), batch_size=BS, epochs=EPOCHS, verbose=1)

# Evaluate the network
print("[INFO] Evaluating the network...")
predictions = model.predict(testData)
print(classification_report(testLabels.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

# Serialize the model to disk
print("[INFO] Serializing the digit model...")
model.save(args["model"], save_format="h5")

