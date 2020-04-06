# -----------------------------
#   USAGE
# -----------------------------
# python blur_face.py --image examples/adrian.jpg --face face_detector --method simple
# python blur_face.py --image examples/adrian.jpg --face face_detector --method pixelated

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.face_blurring import anonymize_face_simple, anonymize_face_pixel_rate
import numpy as np
import argparse
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-f", "--face", required=True, help="path to face detector model directory")
ap.add_argument("-m", "--method", type=str, default="simple", choices=["simple", "pixelated"],
                help="face blurring/anonymizing method")
ap.add_argument("-b", "--blocks", type=int, default=20, help="# of blocks for the pixelated blurring method")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the input image from disk, clone it, and grab the image spatial dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# Construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the face detections
print("[INFO] Computing face detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    # Extract the confidence (i.e, probability) associated with the detection
    confidence = detections[0, 0, i, 2]
    # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
    if confidence > args["confidence"]:
        # Compute the (x, y) coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # Extract the face ROI
        face = image[startY:endY, startX:endX]
        # Employ the "chosen" face blurring method
        if args["method"] == "simple":
            face = anonymize_face_simple(face, factor=3.0)
        # Otherwise, employ the "other" face blurring method
        else:
            face = anonymize_face_pixel_rate(face, blocks=args["blocks"])
        # Store the blurred face in the output image
        image[startY:endY, startX:endX] = face

# Display the original image and the output blurred face image side by side
output = np.hstack([orig, image])
cv2.imshow("Output", output)
cv2.waitKey(0)
