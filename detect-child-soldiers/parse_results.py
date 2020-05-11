# -----------------------------
#   USAGE
# -----------------------------
# python parse_results.py --ages output/ages.csv --camo output/camo.csv

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.helpers import anonymize_face_pixelate
import numpy as np
import argparse
import imutils
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--ages", required=True, help="path to input ages CSV file")
ap.add_argument("-c", "--camo", required=True, help="path to input camo CSV file")
args = vars(ap.parse_args())

# Load the contents of the ages and camo CSV files
ageRows = open(args["ages"]).read().strip().split("\n")
camoRows = open(args["camo"]).read().strip().split("\n")

# Initialize two dictionaries, one to store the age results and the other to store the camo results, respectively
ages = {}
camo = {}

# Loop over the age rows
for row in ageRows:
    # Parse the row
    row = row.split(",")
    imagePath = row[0]
    bbox = [int(x) for x in row[1:5]]
    age = row[5]
    ageProb = float(row[6])
    # Construct a tuple that consists of the bounding box coordinates, the age and the age probability
    t = (bbox, age, ageProb)
    # Update the age dictionary to use the image path as the key and the detection information as a tuple
    l = ages.get(imagePath, [])
    l.append(t)
    ages[imagePath] = l

# Loop over the camo rows
for row in camoRows:
    # Parse the row
    row = row.split(",")
    imagePath = row[0]
    camoProb = float(row[1])
    # Update the camo dictionary to use the image path as the key and the camouflage probability as the value
    camo[imagePath] = camoProb

# Find all the image paths that exists in both the age dictionary and the camo dictionary
inter = sorted(set(ages.keys()).intersection(camo.keys()))

# Loop over all the image paths in the intersection
for imagePath in inter:
    # Load the input image and grab its dimensions
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # If the width is greater than the height, resize along the width dimension
    if w > h:
        image = imutils.resize(image, width=600)
    # Otherwise, resize the image along the height
    else:
        image = imutils.resize(image, height=600)
    # Compute the resize ratio, which is the ratio between the *new* image dimensions to the *old* image dimensions
    ratio = image.shape[1] / float(w)
    # Loop over the age predictions for this particular image
    for (bbox, age, ageProb) in ages[imagePath]:
        # Extract the bounding box coordinates of the face detection
        bbox = [int(x) for x in np.array(bbox) * ratio]
        (startX, startY, endX, endY) = bbox
        # Anonymize the face
        face = image[startY:endY, startX:endX]
        face = anonymize_face_pixelate(face, blocks=5)
        image[startY:endY, startX:endX] = face
        # Set the color for the annotation to *green*
        color = (0, 255, 0)
        # Override the color to *red* they are potential child soldier
        if age in ["(0-2)", "(4-6)", "(8-12)", "(15-20)"]:
            color = (0, 0, 255)
        # Draw the bounding box of the face along with the associated predicted age
        text = "{}: {:.2f}%".format(age, ageProb * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    # Draw the camouflage prediction probability on the image
    label = "camo: {:.2f}%".format(camo[imagePath] * 100)
    cv2.rectangle(image, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
