# -----------------------------
#   USAGE
# -----------------------------
# python search.py --model output/autoencoder.h5 --index output/index.pickle

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def euclidean(a, b):
    # Compute and return the euclidean distance between two vectors
    return np.linalg.norm(a - b)


def perform_search(queryFeatures, index, maxResults=64):
    # Initialize the list of results
    results = []
    # Loop over the index
    for i in range(0, len(index["features"])):
        # Compute the euclidean distance between the query features and the features for the current image in the index,
        # then update the results list with a 2-tuple consisting of the computed distance and the index of the image
        d = euclidean(queryFeatures, index["features"][i])
        results.append((d, i))
    # Sort the results and grab the top ones
    results = sorted(results)[:maxResults]
    # Return the list of results
    return results


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained autoencoder")
ap.add_argument("-i", "--index", type=str, required=True, help="path to features index file")
ap.add_argument("-s", "--sample", type=int, default=10, help="number of testing queries to perform")
args = vars(ap.parse_args())

# Load the MNIST dataset
print("[INFO] Loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()

# Add a channel dimension to every image in the dataset, then scale the pixel intensities to the range[0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Load the autoencoder model and index from the disk
print("[INFO] Loading the autoencoder model and index from disk...")
autoencoder = load_model(args["model"])
index = pickle.loads(open(args["index"], "rb").read())

# Create the encoder model which only consists of the encoder portion of the autoencoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("encoded").output)

# Quantify the contents of the input testing images using the encoder
print("[INFO] Encoding testing images...")
features = encoder.predict(testX)

# Randomly sample a set of testing query images indexes
queryIdxs = list(range(0, testX.shape[0]))
queryIdxs = np.random.choice(queryIdxs, size=args["sample"], replace=False)

# Loop over the testing indexes
for i in queryIdxs:
    # Take the features for the current image, find all similar images in the dataset and then initialize the
    # list of result images
    queryFeatures = features[i]
    results = perform_search(queryFeatures, index, maxResults=225)
    images = []
    # Loop over the results
    for (d, j) in results:
        # Grab the result image, convert it back to the range [0, 255] and then update the image list
        image = (trainX[j] * 255).astype("uint8")
        image = np.dstack([image] * 3)
        images.append(image)
    # Display the query image
    query = (testX[i] * 255).astype("uint8")
    cv2.imshow("Query", query)
    # Build a montage from the results and display it
    montage = build_montages(images, (28, 28), (15, 15))[0]
    cv2.imshow("Results", montage)
    cv2.waitKey(0)
