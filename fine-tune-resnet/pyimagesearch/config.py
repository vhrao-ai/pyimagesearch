# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os

# Initialize the path to the original input directory for the images
ORIG_INPUT_DATASET = "8k_normal_vs_camouflage_clothes_images"

# Initialize the base path to the new directory that will contain the images
# after computing the training and testing splits
BASE_PATH = "camo_not_camo"

# Derive the training, validation and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# Define the amount of data that will be used for training
TRAIN_SPLIT = 0.75

# Define the amount of data that will be used for validation
VAL_SPLIT = 0.1

# Define the names of the classes
CLASSES = ["camouflage_clothes", "normal_clothes"]

# Initialize the initial learning rate, batch size and number of epochs to train for
INIT_LR = 1e-4
BS = 32
NUM_EPOCHS = 20

# Define the path to the serialized output model after training
MODEL_PATH = "camo_detector.model"
