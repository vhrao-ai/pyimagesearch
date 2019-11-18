# ----------------------
#   USAGE
# ----------------------
# python predict_fire.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from keras.models import load_model
from imutils import paths
import numpy as np
import config_params
import imutils
import random
import cv2
import os

# Load the trained model from disk
print("[INFO] Loading the training mode...")
model = load_model(config_params.MODEL_PATH)

# Grab the paths to the fire and non-fire images, respectively
print("[INFO] Predicting...")
firePaths = list(paths.list_images(config_params.FIRE_PATH))
nonFirePaths = list(paths.list_images(config_params.NON_FIRE_PATH))

# Combine the two image paths lists, randomly shuffle them and sample them
imagePaths = firePaths + nonFirePaths
random.shuffle(imagePaths)
imagePaths = imagePaths[:config_params.SAMPLE_SIZE]

# Loop over the sample image paths
for (i, imgPath) in enumerate(imagePaths):
    # Load the image and clone it
    img = cv2.imread(imgPath)
    output = img.copy()
    # Resize the input image to be a fixed 128x128 pixels, ignoring aspect ratio
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    # Make predictions on the image
    preds = model.predict(np.expand_dims(img, axis=0))[0]
    j = np.argmax(preds)
    label = config_params.CLASSES[j]
    # Draw the activity on the output frame
    text = label if label == "Non-Fire" else "WARNING! Fire!"
    output = imutils.resize(output, width=500)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    # Write the output image to disk
    filename = "{}.png".format(i)
    p = os.path.sep.join([config_params.OUTPUT_IMAGE_PATH, filename])
    cv2.imwrite(p, output)
