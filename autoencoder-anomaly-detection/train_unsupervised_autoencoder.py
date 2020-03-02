# ------------------------
#   USAGE
# ------------------------
# python train_unsupervised_autoencoder.py --dataset output/images.pickle --model output/autoencoder.model

# ------------------------
#   IMPORTS
# ------------------------
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# Import the necessary packages
from pyimagesearch.convautoencoder import ConvAutoEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2


# ------------------------
#   FUNCTIONS
# ------------------------
def build_unsupervised_dataset(data, labels, valid_label=1, anomaly_label=3, contam=0.01, seed=42):
    # Grab all indexes of the supplied class label that are *truly* that particular label, then grab the indexes
    # of the image labels that will serve as the "anomalies"
    valid_idxs = np.where(labels == valid_label)[0]
    anomaly_idxs = np.where(labels == anomaly_label)[0]
    # Randomly shuffle both sets of indexes
    random.shuffle(valid_idxs)
    random.shuffle(anomaly_idxs)
    # Compute the total number of anomaly data points to select
    i = int(len(valid_idxs) * contam)
    anomalyIdxs = anomaly_idxs[:i]
    # Use NumPy array indexing to extract both the valid images and "anomaly" images
    valid_images = data[valid_idxs]
    anomaly_images = data[anomalyIdxs]
    # Stack the valid images and anomaly images together to form a single data matrix and then shuffle the rows
    images = np.vstack([valid_images, anomaly_images])
    np.random.seed(seed)
    np.random.shuffle(images)
    # Return the set of images
    return images


def visualize_predictions(decoded, gt, samples=10):
    # Initialize the list of output images
    outputs = None
    # loop over the number of output samples
    for i in range(0, samples):
        # Grab the original image and reconstructed image
        original = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")
        # Stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])
        # If the outputs array is empty, initialize it as the current side-by-side image display
        if outputs is None:
            outputs = output
        # Otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])
    # Return the output images
    return outputs


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to output dataset file")
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained autoencoder")
ap.add_argument("-v", "--vis", type=str, default="recon_vis.png",
                help="path to output reconstruction visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output plot file")
args = vars(ap.parse_args())

# Initialize the number of epochs to train for, the initial learning rate and the batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 32

# Load the MNIST dataset
print("[INFO] Loading the MNIST dataset...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# Build the unsupervised dataset of images with a small amount of contamination (i.e, anomalies) added into it
print("[INFO] Creating unsupervised dataset...")
images = build_unsupervised_dataset(trainX, trainY, valid_label=1, anomaly_label=3, contam=0.01)

# Add a channel dimension to every image in the dataset, then scale the pixel intensities to the range [0, 1]
images = np.expand_dims(images, axis=-1)
images = images.astype("float32") / 255.0

# Construct the training and testing split
(trainX, testX) = train_test_split(images, test_size=0.2, random_state=42)

# Construct the convolutional autoencoder
print("[INFO] Building the  convolutional autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoEncoder.build(28, 28, 1)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)

# Train the convolutional autoencoder
H = autoencoder.fit(trainX, trainX, validation_data=(testX, testX), epochs=EPOCHS, batch_size=BS)

# Use the convolutional autoencoder to make predictions on the testing images, construct the visualization and save it
print("[INFO] Making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite(args["vis"], vis)

# Construct the plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Serialize the image data to disk
print("[INFO] Saving image data...")
f = open(args["dataset"], "wb")
f.write(pickle.dumps(images))
f.close()

# Serialize the autoencoder model to disk
print("[INFO] Saving autoencoder...")
autoencoder.save(args["model"], save_format="h5")
