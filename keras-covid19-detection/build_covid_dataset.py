# -----------------------------
#   USAGE
# -----------------------------
# python build_covid_dataset.py --covid covid-chestxray-dataset --output dataset/covid


# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import pandas as pd
import argparse
import shutil
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--covid", required=True, help="Path to base directory for COVID-19 dataset")
ap.add_argument("-o", "--output", required=True, help="Path to directory where 'normal' images will be stored")
args = vars(ap.parse_args())

# Construct the the path to the metadata CSV file and load it
csv_path = os.path.sep.join([args["covid"], "metadata.csv"])
df = pd.read_csv(csv_path)

# Loop over the rows of the COVID-19 data frame
for (i, row) in df.iterrows():
    # If (1) the current case is not COVID-19 or (2) this is not a 'PA' view, then ignore the row
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue
    # Build the path to the input image file
    img_path = os.path.sep.join([args["covid"], "images", row["filename"]])
    # If the input image file does not exist (there are some errors in the COVID-19 metadata file), ignore row
    if not os.path.exists(img_path):
        continue
    # Extract the filename from the image path and construct the path to the copied image file
    filename = row["filename"].split(os.path.sep)[-1]
    output_path = os.path.sep.join([args["output"], filename])
    # Copy the image
    shutil.copy2(img_path, output_path)
