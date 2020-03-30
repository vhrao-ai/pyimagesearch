# -----------------------------
#   USAGE
# -----------------------------
# python train_conv_autoencoder.py --model output/autoencoder.h5 --vis output/recon_vis.png --plot output/plot.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# Import the necessary packages
from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def visualize_predictions(decoded, gt, samples=10):
    # Initialize the list of output images
    outputs = None
    # Loop over the number of output samples
    for i in range(0, samples):
        # Grab the original image and reconstructed image
        original = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")
        # Stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])
        # If the outputs array is empty, initialize it as the current side-by-side image display
        if outputs is None:
            outputs = output
        # Otherwise, vertically stack on the outputs
        else:
            outputs = np.vstack([outputs, output])
    # Return the output images
    return outputs


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained autoencoder")
ap.add_argument("-v", "--vis", type=str, default="recon_vis.png",
                help="path to output reconstruction visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output plot file")
args = vars(ap.parse_args())

# Initialize the number of epochs to train for, initial learning rate and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 32

# Load the MNIST dataset
print("[INFO] Loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()

# Add a channel dimension to every image in the dataset, then scale the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Construct the convolutional autoencoder
print("[INFO] Building autoencoder...")
autoencoder = ConvAutoencoder.build(28, 28, 1)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)

# Train the convolutional autoencoder
print("[INFO] Training autoencoder...")
H = autoencoder.fit(trainX, trainX, validation_data=(testX, testX), epochs=EPOCHS, batch_size=BS)

# Use the convolutional autoencoder to make predictions on the testing images, construct the visualization,
# and then save it to disk
print("[INFO] Making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite(args["vis"], vis)

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

# Serialize the autoencoder model to disk
print("[INFO] Saving autoencoder model...")
autoencoder.save(args["model"], save_format="h5")