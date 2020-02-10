# -----------------------------
#   USAGE
# -----------------------------
# python mask_rcnn_segmentation.py --input ../example_videos/dog_park.mp4 --output ../output_videos/mask_rcnn_dog_park.avi --display 0 --mask-rcnn mask-rcnn-coco
# python mask_rcnn_segmentation.py --input ../example_videos/dog_park.mp4 --output ../output_videos/mask_rcnn_dog_park.avi --display 0 --mask-rcnn mask-rcnn-coco --use-gpu 1

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
ap.add_argument("-m", "--mask-rcnn", required=True, help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-u", "--use-gpu", type=bool, default=0, help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# Load the COCO class labels to the Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"], "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"], "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# Load the Mask R-CNN trained on the COCO dataset (90 classes) from disk
print("[INFO] Loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# Check if we are going to use GPU
if args["use_gpu"]:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize the video stream and pointer to output video file, then start the FPS timer
print("[INFO] Accessing the video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = FPS().start()

# Loop over frames from the video file stream
while True:
    # Read the next frame from the file
    (grabbed, frame) = vs.read()
    # If the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    # Construct a blob from the input frame and then perform a forward pass of the Mask R-CNN,
    # giving us (1) the bounding box coordinates of the objects in the image along with (2) the
    # pixel-wise segmentation for each specific object
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    # Loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # Extract the class ID of the detection along with the confidence (i.e., probability) associated with the
        # prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # Scale the bounding box coordinates back relative to the size of the frame and
            # then compute the width and the height of the bounding box
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            # Extract the pixel-wise segmentation for the object, resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > args["threshold"])
            # Extract the ROI of the image but *only* extracted the masked region of the ROI
            roi = frame[startY:endY, startX:endX][mask]
            # Grab the color used to visualize this particular class, then create a transparent overlay
            # by blending the color with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
            # Store the blended ROI in the original frame
            frame[startY:endY, startX:endX][mask] = blended
            # Draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # Draw the predicted label and associated probability of the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Check to see if the output frame should be displayed to the screen
    if args["display"] > 0:
        # Show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # If an output video file path has been supplied and the video writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # Initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # If the video writer is not None, write the frame to the output video file
    if writer is not None:
        writer.write(frame)
    # Update the FPS counter
    fps.update()
# Stop the timer and display FPS information
fps.stop()
print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))