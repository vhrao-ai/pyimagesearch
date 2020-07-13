# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os

# Define the base path to the *original* input dataset and then use the base path
# to derive the image and annotations directories
ORIG_BASE_PATH = "raccoons"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTATIONS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

# Define the base path to the *new* dataset after running the dataset builder scripts and then use the base path to
# derive the paths to the output class label directories
BASE_PATH = "dataset"
POSITIVE_PATH = os.path.sep.join([BASE_PATH, "raccoon"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_raccoon"])

# Define the number max proposals used when running the selective search for:
# (1) Gathering the training data and (2) Performing the inference process
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# Define the maximum number of positive and negative images to be generated from each image
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# Initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# Define the path to the output network model and the label binarizer
MODEL_PATH = "raccoon_detector.h5"
ENCODER_PATH = "label_encoder.pickle"

# Define the minimum probability required for a positive prediction (used to filter out false-positive predictions)
MIN_PROB = 0.99

