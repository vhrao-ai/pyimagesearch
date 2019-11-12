# ------------------------
#   USAGE
# ------------------------
# python predict.py --input terrific_natural_disasters_compilation.mp4 --output output/natural_disasters.avi

# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from keras.models import load_model
from collections import deque
from .pyimagesearch import config
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to our input video")
ap.add_argument("-o", "--output", required=True, help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128, help="size of queue for averaging")
ap.add_argument("-d", "--display", type=int, default=-1,
                help="whether or not output frame should be displayed to screen")
args = vars(ap.parse_args())

# Load the trained model from the disk
print("[INFO] Loading model and Label Binarizer...")
model = load_model(config.MODEL_PATH)

# Initialize the predictions queue
Q = deque(maxlen=args["size"])

# Initialize the video stream, pointer to the output video file and frame dimensions
print("[INFO] Processing the video...")
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# Loop over the frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # clone the output frame, then convert it from BGR to RGB and resize the frame to a fixed 224x224
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32")
    # make the predictions on the frame and then update the predictions queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    # perform prediction averaging over the current history of previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = config.CLASSES[i]
    # draw the activity on the output frame
    text = "Activity: {}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    # check if the video writer is None
    if writer is None:
        # initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
    # write the output frame to disk
    writer.write(output)
    # check to see if we should display the output frame to our screen
    if args["display"] > 0:
        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break
# release the file pointers
print("[INFO] Cleaning up...")
writer.release()
vs.release()