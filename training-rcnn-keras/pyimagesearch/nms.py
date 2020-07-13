# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np

# -----------------------------
#   FUNCTIONS
# -----------------------------
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # If the bounding boxes are integers, convert them into floats
    # -- this is very important since we will be doing a lot of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # Initialize the list of picked indexes
    pick = []
    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # Compute the area of the bounding boxes and grab the indexes to sort (in the case there are no probabilities,
    # simply sort on the bottom left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2
    # If probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs
    # Sort the indexes
    idxs = np.argsort(idxs)
    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box coordinates
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # Compute the width and the height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # Delete all the indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # Return the indexes of only the bounding boxes to keep
    return pick

