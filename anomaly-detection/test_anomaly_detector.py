# -----------------------------
#   USAGE
# -----------------------------
# python test_anomaly_detector.py --model trained_anomaly_detector.model

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.features import quantify_image
import argparse
import pickle
import cv2
import glob


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained anomaly detection model")
args = vars(ap.parse_args())

# Load the image path
img_path = 'examples/*.jpg'

# Load the anomaly detection model
print("[INFO] Loading Anomaly Detection Model...")
model = pickle.loads(open(args["model"], "rb").read())


# Check each one of the image for anomalies
for i, filename in enumerate(glob.iglob(img_path)):
    name_file = filename.split('\\')[-1].split('.')[0]
    print('Reading image: {}.jpg'.format(name_file))
    # Load the input image, convert it to the HSV color space, and quantify the image
    # in the same way as the training process
    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = quantify_image(hsv, bins=(3, 3, 3))
    # Use the anomaly detector model and the extracted features to determine if the example image is an anomaly or not
    preds = model.predict([features])[0]
    label = "anomaly" if preds == -1 else "normal"
    color = (0, 0, 255) if preds == -1 else (0, 255, 0)
    # Draw the predicted label text on the original image
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Display the image
    cv2.imshow("Output", img)
    cv2.waitKey(0)