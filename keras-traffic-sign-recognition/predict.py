# ------------------------
#   USAGE
# ------------------------
# python predict.py --model output/trafficsignnet.model --images gtsrb-german-traffic-sign/Test --examples examples

# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to pre-trained traffic sign recognizer")
ap.add_argument("-i", "--images", required=True, help="path to testing directory containing images")
ap.add_argument("-e", "--examples", required=True, help="path to output examples directory")
args = vars(ap.parse_args())

# Load the traffic sign recognizer model
print("[INFO] Loading the model...")
model = load_model(args["model"])

# Load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

# Grab the paths to the input images, shuffle them and grab a sample
print("[INFO] Predicting...")
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:25]

# Loop over the image paths
for (i, imgPath) in enumerate(imagePaths):
    # Load the image, resize it to 32x32 pixels and then apply the Contrast Limited Adaptive Histogram Equalization
    # just like we did during the training process
    img = io.imread(imgPath)
    img = transform.resize(img, (32, 32))
    img = exposure.equalize_adapthist(img, clip_limit=0.1)
    # Preprocess the image by scaling it to the range [0, 1]
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    # Make predictions using the traffic sign recognizer CNN
    preds = model.predict(img)
    j = preds.argmax(axis=1)[0]
    label = labelNames[j]
    # Load the image using OpenCV, resize it, and draw the label on it
    img = cv2.imread(imgPath)
    img = imutils.resize(img, width=128)
    cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # Save the image to disk
    p = os.path.sep.join([args["examples"], "{}.png".format(i)])
    cv2.imwrite(p, img)


