# ------------------------
#   USAGE
# ------------------------
# python find_anomalies.py --dataset output/images.pickle --model output/autoencoder.model

# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input image dataset file")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained autoencoder")
ap.add_argument("-q", "--quantile", type=float, default=0.999, help="q-th quantile used to identify outliers")
args = vars(ap.parse_args())

# Load the model and image data from disk
print("[INFO] Loading autoencoder and image data...")
autoencoder = load_model(args["model"])
images = pickle.loads(open(args["dataset"], "rb").read())

# Make predictions on the image data and initialize the list of reconstruction errors
decoded = autoencoder.predict(images)
errors = []

# Loop over all of the original images and their corresponding reconstructions
for (image, recon) in zip(images, decoded):
    # Compute the mean squared error between the ground-truth image and the reconstructed image
    # then add it to error list
    mse = np.mean((image - recon) ** 2)
    errors.append(mse)

# Compute the q-th quantile of the erros which serves as the threshold that will be used to identify the anomalies
# -- Any data point that the model reconstructed > threshold error will be marked as an outlier
thresh = np.quantile(errors, args["quantile"])
idxs = np.where(np.array(errors) >= thresh)[0]
print("[INFO] MSE Threshold: {}".format(thresh))
print("[INFO] {} Outliers Found".format(len(idxs)))

# Initialize the outputs array
outputs = None

# Loop over the indexes of the images with a high mean squared error term
for i in idxs:
    # Grab the original image and reconstructed image
    original = (images[i] * 255).astype("uint8")
    recon = (decoded[i] * 255).astype("uint8")
    # Stack the original and reconstructed image side-by-side
    output =np.hstack([original, recon])
    # If the outputs array is empty, initialize it as the current side-by-side image display
    if outputs is None:
        outputs = output
    # Otherwise, vertically stack the outputs
    else:
        outputs = np.vstack([outputs, output])

# Show the output visualization
cv2.imshow("Output", outputs)
cv2.waitKey(0)