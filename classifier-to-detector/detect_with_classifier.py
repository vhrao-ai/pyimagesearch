# -----------------------------
#   USAGE
# -----------------------------
# python detect_with_classifier.py --image images/stingray.jpg --size "(300, 150)"
# python detect_with_classifier.py --image images/hummingbird.jpg --size "(250, 250)"
# python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)" --min-conf 0.95

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from pyimagesearch.detection_helpers import sliding_window
from pyimagesearch.detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)", help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9, help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1, help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

# Initialize the variables that are going to be used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

# Load the network weights from disk
print("[INFO] Loading network...")
model = ResNet50(weights="imagenet", include_top=True)

# Load the input image from disk, resize it such that it has the supplied width, and then grab the dimensions
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]

# Initialize the image pyramid
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

# Initialize two lists, one to hold the ROIs generated from the image
# Pyramid and sliding window, and another list used to store (x,y) coordinates
# where the ROI was in the original image
rois = []
locs = []

# Time how long it takes to loop over the image pyramid layers and sliding window locations
start = time.time()

# Loop over the image pyramid
for image in pyramid:
    # Determine the scale factor between the original image dimensions and the current layer of the pyramid
    scale = W/float(image.shape[1])
    # For each layer of the image pyramid, loop over the sliding window locations
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # Scale the (x, y) coordinates of the ROI with respect to the original image dimensions
        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)
        # Take the ROI and pre-process it in order to classify later the region using Keras/Tensorflow
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        # Update the list of ROIs and associated coordinates
        rois.append(roi)
        locs.append((x, y, x+w, y+h))
        # Check to see if we are visualizing each one of the sliding windows in the image pyramid
        if args["visualize"] > 0:
            # Clone the original image and then draw a bounding box surrounding the current region
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Show the visualization and current ROI
            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)

# Show how long it too to loop over the image pyramid layers and the sliding window locations
end = time.time()
print("[INFO] Looping over pyramid/windows took {:.5f} seconds".format(end-start))

# Convert the ROIs to a Numpy array
rois = np.array(rois, dtype="float32")

# Classify each of the proposal ROIs using ResNet and then show how long the classifications took
print("[INFO] Classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] Classifying ROIs took {:.5f} seconds".format(end-start))

# Decode the predictions and initialize a dictionary which maps class labels
# (keys) to any ROIs associated with that label (values)
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

# Loop over the predictions
for (i, p) in enumerate(preds):
    # Grab the prediction information for the current ROI
    (imagenetID, label, prob) = p[0]
    # Filter out weak detections by ensuring the predicted probability is greater than the minimum probability
    if prob >= args["min_conf"]:
        # Grab the bounding box associated with the prediction and convert the coordinates
        box = locs[i]
        # Grab the list of predictions for the label and add the bounding box and probability to the list
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

# Loop over the labels for each of detected objects in the image
for label in labels.keys():
    # Clone the original in order to draw on it
    print("[INFO] Showing results for '{}'".format(label))
    clone = orig.copy()
    # Loop over all bounding boxes for the current label
    for (box, prob) in labels[label]:
        # Draw the bounding box on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # Show the results before applying non-maximum suppression,
    # clone the image again in order to display the results after applying non-maximum suppression
    cv2.imshow("Before", clone)
    clone = orig.copy()
    # Extract the bounding boxes and associated prediction probabilities, then apply non-maximum suppression
    boxes = np.array([p[0] for p in labels[label]])
    prob = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, prob)
    # Loop over all bounding boxes that we kept after applying non-maximum suppression
    for (startX, startY, endX, endY) in boxes:
        # Draw the bounding box and label on the image
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    # Show the output after applying non-maximum suppression
    cv2.imshow("After", clone)
    cv2.waitKey(0)


