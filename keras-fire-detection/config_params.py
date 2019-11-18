# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os

# Initialize the path to the fire and non-fire dataset directories
FIRE_PATH = os.path.sep.join(["Robbery_Accident_Fire_Database2", "Fire"])
NON_FIRE_PATH = "spatial_envelope_256x256_static_8outdoorcategories"

# Initialize the class labels in the dataset
CLASSES = ["Non-Fire", "Fire"]

# Define the size of the training and testing split
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

# Define the initial learning rate, batch size, and number of epochs
INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50

# Set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "fire_detection.model"])

# Define the path to the output learning rate finder plot and training history plot
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])

# Define the path to the output directory that will store our final
# output with labels/annotations along with the number of iamges to sample
OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])
SAMPLE_SIZE = 50