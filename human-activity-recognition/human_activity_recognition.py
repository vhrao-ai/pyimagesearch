# -----------------------------
#   USAGE
# -----------------------------
# python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt
# --input videos/example_activities.mp4
# python human_activity_recognition.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True, help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="", help="optional path to video file")
args = vars(ap.parse_args())

# Load the contents of the class labels file, then define the sample (i.e., # of frames for classification) and sample
# size (i.e., the spatial dimensions of the frame)
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# Load the human activity recognition model
print("[INFO] Loading the human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

# Grab the pointer to the input video stream
print("[INFO] Accessing the video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

# Loop until we explicity break from it
while True:
    # Initialize the batch of frames that will be passed through the model
    frames = []
    # Loop over the number of required sample frames
    for i in range(0, SAMPLE_DURATION):
        # Read a frame from the video stream
        (grabbed, frame) = vs.read()
        # If the frame was not grabbed then we've reached the end of the video stream so exit the script
        if not grabbed:
            print("[INFO] No frame read from the stream - Exiting...")
            sys.exit(0)
        # Otherwise, the frame was read so resize it and add it to the frames list
        frame = imutils.resize(frame, width=400)
        frames.append(frame)
    # Now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                  swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    # Pass the blob through the network to obtain our human activity recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]
    # Loop over the frames
    for frame in frames:
        # Draw the predicted activity on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Display the frame to our screen
        cv2.imshow("Activity Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        # If the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break
