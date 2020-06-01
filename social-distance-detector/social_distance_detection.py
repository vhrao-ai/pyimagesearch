# -----------------------------
#   USAGE
# -----------------------------
# python social_distance_detection.py --input pedestrians.mp4
# python social_distance_detection.py --input pedestrians.mp4 --output output.avi

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# Load the COCO class labels that the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# Load the YOLO object detector trained on the COCO dataset (80 classes)
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Check whether or not GPU is going to be used
if config.USE_GPU:
    # Set CUDA as the preferable backend and target
    print("[INFO] Setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Determine only the names from the output layer that YOLO needs
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video stream and pointer to the output video file
print("[INFO] Accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# Loop over the frames from the video stream
while True:
    # Read the next frame from the file
    (grabbed, frame) = vs.read()
    # Grab frames from the video stream until we have reached the end frame
    if not grabbed:
        break
    # Resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    # Initialize the set of indexes that violate the minimum social distance
    violate = set()
    # Ensure that there at least two people detections (required in order to
    # compute the pairwise distance maps)
    if len(results) >= 2:
        # Extract all centroids from the results and compute the Euclidean distance between all pairs of centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        # Loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # Check to see if the distance between any two centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # Update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)
        # Loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # Extract the bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)
            # If the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)
            # Draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        # Draw the total number of social distancing violations on the output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        # Check to see if the output frame should be displayed in the screen
        if args["display"] > 0:
            # Show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # If the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # If an output video file path has been supplied and the video writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # Initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
        # If the video writer is not None, write the frame to the output video file
        if writer is not None:
            writer.write(frame)

