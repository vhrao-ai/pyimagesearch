# -----------------------------
#   USAGE
# -----------------------------
# python detect_and_remove.py --dataset dataset
# python detect_and_remove.py --dataset dataset --remove 1

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os


# -----------------------------
#   FUNCTIONS
# -----------------------------
def diff_hash(image, hashSize=8):
    # Convert the image to grayscale and resize it adding a single column (width)
    # in order to compute the horizontal gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize+1, hashSize))
    # Compute the (relative) horizontal gradient between the adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
                help="Whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())

# Grab the paths to all images in the input dataset directory and then initialize the hashes dictionary
print("[INFO] Computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))
hashes = {}

# Loop over the image paths
for imagePath in imagePaths:
    # Load the input image and compute the hash
    image = cv2.imread(imagePath)
    h = diff_hash(image)
    # Grab all the images with that hash, add the current image path to it
    # and store the list back in the hashes dictionary
    p = hashes.get(h, [])
    p.append(imagePath)
    hashes[h] = p

# Loop over the image hashes
for (h, hashedPaths) in hashes.items():
    # Check to see if there is more than one image with the same hash
    if len(hashedPaths) > 1:
        # Check to see if this is a dry run
        if args["remove"] <= 0:
            # Initialize a montage to store all images with the same hash
            montage = None
            # Loop over all image paths with the same hash
            for p in hashedPaths:
                # Load the input image and resize it to a fixed width and height
                image = cv2.imread(p)
                image = cv2.resize(image, (150, 150))
                # If the montage is None, initialize it
                if montage is None:
                    montage = image
                # Otherwise, stack the images horizontally
                else:
                    montage = np.hstack([montage, image])
            # Show the montage for the hash
            print("[INFO] Hash: {}".format(h))
            cv2.imshow("Montage", montage)
            cv2.waitKey(0)
        # Otherwise, remove the duplicated images
        else:
            # Loop over all image paths with the same hash except for the first image in the list
            # (in order to keep only one copy of the duplicated images, in other words keep only the original copy)
            for p in hashedPaths[1:]:
                os.remove(p)

