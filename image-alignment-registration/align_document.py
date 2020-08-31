# -----------------------------
#   USAGE
# -----------------------------
# python align_document.py --template form_w4.png --image scans/scan_01.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.alignment.align_images import align_images
import numpy as np
import argparse
import imutils
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True, help="Path to input template image")
args = vars(ap.parse_args())

# Load the input image and template from disk
print("[INFO] Loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# Align the images
print("[INFO] Aligning images...")
aligned = align_images(image, template, debug=True)

# Resize both the aligned and template images in order to visualize them
aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

# The first output visualization of the image alignment will be side by side comparison of the output aligned image
# and the template image
stacked = np.hstack([aligned, template])

# The second output visualization will be "overlaying" the aligned image on the template image, that way it is possible
# to obtain an idea on how good the image alignment is
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

# Show the two output image alignment visualization results side by side
cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)
