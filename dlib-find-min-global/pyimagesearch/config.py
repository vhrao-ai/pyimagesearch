# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
import os

# Define the path to the training and testing XML files
TRAIN_PATH = os.path.join("ibug_300W_large_face_landmark_dataset", "labels_ibug_300W_train_eyes.xml")
TEST_PATH = os.path.join("ibug_300W_large_face_landmark_dataset", "labels_ibug_300W_test_eyes.xml")

# Define the path to the temporary model file
TEMP_MODEL_PATH = "temp.dat"

# Define the number of threads/cores we'll be using when training the shape predictor models
PROCS = -1

# define the maximum number of trials we'll be performing when tuning the shape predictor hyperparameters
MAX_FUNC_CALLS = 100
