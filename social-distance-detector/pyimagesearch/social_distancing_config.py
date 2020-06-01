# Base path to YOLO directory
MODEL_PATH = "yolo-coco"

# Initialize minimum probability to filter weak detections along with the threshold
# when applying non-maximum suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# Boolean indicating if NVIDIA CUDA should be used or not
USE_GPU = False

# Define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE = 50
