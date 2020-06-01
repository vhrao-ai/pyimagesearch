# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def detect_people(frame, net, ln, personIdx=0):
    # Grab the dimensions of the frame and initialize the list of results
    (H, W) = frame.shape[:2]
    results = []
    # Construct a blob from the input frame and then perform a forward pass of the YOLO object detector,
    # giving the corresponding bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # Initialize the list of detected bounding boxes, centroids, and confidences respectively
    boxes = []
    centroids = []
    confidences = []
    # Loop over each one of the layer outputs
    for output in layerOutputs:
        # Loop over each one of the detections
        for detection in output:
            # Extract the class ID and confidence (i.e, probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # Filter detections by:
            # (1) ensuring that the object detected was a person
            # (2) ensuring that the minimum confidence is met.
            if classID == personIdx and confidence > MIN_CONF:
                # Scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y) coordinates of the bounding box followed by boxes width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Use the center (x, y) coordinates to derive the top and left corner fo the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Update the list of bounding box coordinates, centroids and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    # Apply non-maximum suppression to suppress weak detections, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    # Ensure at least one detections exists
    if len(idxs) > 0:
        # Loop over the kept indexes
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # Update the results list that consists of the person prediction probability,
            # the bounding box coordinates, and the centroid
            r = (confidences[i], (x, y, x+w, y+h), centroids[i])
            results.append(r)
    # Return the list of results
    return results

