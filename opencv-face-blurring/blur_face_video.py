# -----------------------------
#   USAGE
# -----------------------------
# python blur_face_video.py --face face_detector --method simple
# python blur_face_video.py --face face_detector --method pixelated

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages

from pyimagesearch.face_blurring import anonymize_face_simple, anonymize_face_pixel_rate
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="path to face detector model directory")
ap.add_argument("-m", "--method", type=str, default="simple", choices=["simple", "pixelated"],
                help="face blurring/anonymizing method")
ap.add_argument("-b", "--blocks", type=int, default=20,
                help="# of blocks for the pixelated blurring method")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # Grab the frame dimensions and construct a blob from those dimensions
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Pass the blob through the network and obtain the face detections
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
            face = frame[startY:endY, startX:endX]
            # Employ the "chosen" face blurring method
            if args["method"] == "simple":
                face = anonymize_face_simple(face, factor=3.0)
            # Otherwise, employ the "other" face blurring method
            else:
                face = anonymize_face_pixel_rate(face, blocks=args["blocks"])
            # Store the blurred face in the output image
            frame[startY:endY, startX:endX] = face
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()