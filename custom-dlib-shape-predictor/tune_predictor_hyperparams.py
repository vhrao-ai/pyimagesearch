# ------------------------
#   USAGE
# ------------------------
# python tune_predictor_hyperparams.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from sklearn.model_selection import ParameterGrid
import multiprocessing
import numpy as np
import random
import time
import dlib
import cv2
import os

# -----------------------------
#   CONFIGURATION PARAMETERS
# -----------------------------
# Define the path to the training and testing XML files
TRAIN_PATH = os.path.join("ibug_300W_large_face_landmark_dataset", "labels_ibug_300W_train_eyes.xml")
TEST_PATH = os.path.join("ibug_300W_large_face_landmark_dataset", "labels_ibug_300W_test_eyes.xml")
# Define the path to the temporary model file
TEMP_MODEL_PATH = "temp.dat"
# Define the path to the output CSV file containing the results of the experiments
CSV_PATH = "trials.csv"
# Define the path to the example image we'll be using to evaluate inference speed using the shape predictor
IMAGE_PATH = "example.jpg"
# Define the number of threads/core we'll be using when training the shape predictor models
PROCS = -1
# Define the maximum number of trials we'll be performing when tuning the shape predictor hyperparameters
MAX_TRIALS = 100


# -----------------------------
#   FUNCTIONS
# -----------------------------
def evaluate_model_acc(xml_path, pred_path):
    # Compute and return the error (lower is better) of the shape predictor over the testing path
    return dlib.test_shape_predictor(xml_path, pred_path)


def evaluate_model_speed(predictor, img_path, tests=10):
    # Initialize the list of timings
    timings = []
    # Loop over the number speeds tests to perform
    for i in range(0, tests):
        # Load the input image and convert it to grayscale
        img = cv2.imread(IMAGE_PATH)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        detector = dlib.get_frontal_face_detector()
        rects = detector(gray, 1)
        start = None
        end = None
        # Ensure at least one face was detected
        if len(rects) > 0:
            # Check the time how long it takes to perform shape prediction using the current shape prediction model
            start = time.time()
            shape = predictor(gray, rects[0])
            end = time.time()
        # Update the timings list
        timings.append(end - start)
    # Compute and return the average over the timings
    return np.average(timings)


# Define the columns of the output CSV file
cols = [
    "tree_depth", "nu", "cascade_depth", "feature_pool_size", "num_test_splits", "oversampling_amount",
    "oversampling_translation_jitter", "inference_speed", "training_time", "training_error", "testing_error",
    "model_size"]

# Open the CSV file for writing and then write the columns as the header of the CSV file
csv = open(CSV_PATH, "w")
csv.write("{}\n".format(",".join(cols)))

# Determine the number of processes/threads to use
procs = multiprocessing.cpu_count()
procs = PROCS if PROCS > 0 else procs

# Initialize the list of dlib shape predictor hyperparameters that we'll be tuning over
hyperparams = {"tree_depth": list(range(2, 8, 2)), "nu": [0.01, 0.1, 0.25], "cascade_depth": list(range(6, 16, 2)),
               "feature_pool_size": [100, 250, 500, 750, 1000], "num_test_splits": [20, 100, 300],
               "oversampling_amount": [1, 20, 40], "oversampling_translation_jitter": [0.0, 0.1, 0.25]
}

# Construct the set of hyperparameter combinations and randomly sample them as trying to test *all* of them
# would be computationally prohibitive
combos = list(ParameterGrid(hyperparams))
random.shuffle(combos)
sampled_combos = combos[:MAX_TRIALS]
print("[INFO] Sampling {} of {} possible combinations".format(len(sampled_combos), len(combos)))

# Loop over the hyperparameter combinations
for (i, p) in enumerate(sampled_combos):
    # Log the experiment number
    print("[INFO] Starting the trial {}/{}...".format(i + 1, len(sampled_combos)))
    # Grab the default options for dlib's shape predictor and then set the values based on
    # the current hyperparameter values
    options = dlib.shape_predictor_training_options()
    options.tree_depth = p["tree_depth"]
    options.nu = p["nu"]
    options.cascade_depth = p["cascade_depth"]
    options.feature_pool_size = p["feature_pool_size"]
    options.num_test_splits = p["num_test_splits"]
    options.oversampling_amount = p["oversampling_amount"]
    options.oversampling_translation_jitter = p["oversampling_translation_jitter"]
    # Tell dlib to be verbose when training and utilize the supplied number of threads when training
    options.be_verbose = True
    options.num_threads = procs
    # Train the model using the current set of hyperparameters
    start = time.time()
    dlib.train_shape_predictor(TRAIN_PATH, TEMP_MODEL_PATH, options)
    trainingTime = time.time() - start
    # Evaluate the model on both the training and testing split
    trainingError = evaluate_model_acc(TRAIN_PATH, TEMP_MODEL_PATH)
    testingError = evaluate_model_acc(TEST_PATH, TEMP_MODEL_PATH)
    # Compute an approximate inference speed using the trained shape predictor
    predictor = dlib.shape_predictor(TEMP_MODEL_PATH)
    inferenceSpeed = evaluate_model_speed(predictor, IMAGE_PATH)
    # Determine the model size
    modelSize = os.path.getsize(TEMP_MODEL_PATH)
    # Build the row of data that will be written to our CSV file
    row = [
        p["tree_depth"],
        p["nu"],
        p["cascade_depth"],
        p["feature_pool_size"],
        p["num_test_splits"],
        p["oversampling_amount"],
        p["oversampling_translation_jitter"],
        inferenceSpeed,
        trainingTime,
        trainingError,
        testingError,
        modelSize,
    ]
    row = [str(x) for x in row]
    # Write the output row to the CSV file
    csv.write("{}\n".format(",".join(row)))
    csv.flush()
    # Delete the temporary shape predictor model
    if os.path.exists(TEMP_MODEL_PATH):
        os.remove(TEMP_MODEL_PATH)
# close the output CSV file
print("[INFO] Cleaning up")