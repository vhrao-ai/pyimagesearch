# ------------------------
#   USAGE
# ------------------------
# python recognize_video.py --detector face_detection_model \
#	--embedding-model face_embedding_model/openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True, help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the serialized face detector from disk
print("[INFO] Loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Load the serialized face embedding model from the disk and set the preferable target to MYRIAD
print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# Initialize the video stream, then allow the camera sensor to warm up
print("[INFO] Starting the video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# Start the FPS throughput estimator
fps = FPS().start()

# Loop over the frame from the video file stream
while True:
    # Grab the frame from the threaded video file stream
    frame = vs.read()
    # Resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the
    # image dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    # Construct the blob from the image
    imgBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0),
                                    swapRB=False, crop=False)
    # Apply OpenCV's deep learning based face detector to localize faces in the input image
    detector.setInput(imgBlob)
    detections = detector.forward()
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > args["confidence"]:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # Ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Construct a blob for the face ROI, then pass the blob through our face embedding model to obtain
            # the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(cv2.resize(face,
                                                        (96, 96)), 1.0 / 255, (96, 96), (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # Perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            # Draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # Update the FPS counter
    fps.update()
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# Stop the timer and display FPS information
fps.stop()
print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()