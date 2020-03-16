# -----------------------------
#   USAGE
# -----------------------------
# python sample_kaggle_dataset.py --kaggle chest_xray --output dataset/normal


# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils import paths
import argparse
import random
import shutil
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--kaggle", required=True, help="Path to base directory of Kaggle X-ray dataset")
ap.add_argument("-o", "--output", required=True, help="Path to directory where 'normal' images will be stored")
ap.add_argument("-s", "--sample", type=int, default=25, help="# of samples to pull from Kaggle dataset")
args = vars(ap.parse_args())

# Grab all the training image paths from the Kaggle X-ray dataset
base_path = os.path.sep.join([args["kaggle"], "train", "NORMAL"])
img_paths = list(paths.list_images(args["kaggle"]))

# Randomly sample the image paths
random.seed(42)
random.shuffle(img_paths)
img_paths = img_paths[:args["sample"]]

# Loop over the image paths
for (i, img_path) in enumerate(img_paths):
    # Extract the filename from the image path and then construct the path to the copied image file
    filename = img_path.split(os.path.sep)[-1]
    output_path = os.path.sep.join([args["output"], filename])
    # Copy the image
    shutil.copy2(img_path, output_path)
