# ------------------------
#   USAGE
# ------------------------
# python train_best_predictor.py --model best_predictor.dat

# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from pyimagesearch import config
import multiprocessing
import argparse
import dlib

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path serialized dlib shape predictor model")
args = vars(ap.parse_args())

# Determine the number of processes/threads to use
procs = multiprocessing.cpu_count()
procs = config.PROCS if config.PROCS > 0 else procs

# Grab the default options for dlib's shape predictor
print("[INFO] Setting shape predictor options...")
options = dlib.shape_predictor_training_options()

# Update the hyperparameters
options.tree_depth = 4
options.nu = 0.1033
options.cascade_depth = 20
options.feature_pool_size = 677
options.num_test_splits = 295
options.oversampling_amount = 29
options.oversampling_translation_jitter = 0
options.feature_pool_region_padding = 0.0975
options.lambda_param = 0.0251

# Tell the dlib shape predictor to be verbose and print out status messages our model trains
options.be_verbose = True

# Number of threads/CPU cores to be used when training -- we default this value to the number of available cores
# on the system, but you can supply an integer value here if you would like
options.num_threads = procs

# Log the training options to the terminal
print("[INFO] Shape predictor options:")
print(options)

# Train the shape predictor
print("[INFO] Training shape predictor...")
dlib.train_shape_predictor(config.TRAIN_PATH, args["model"], options)