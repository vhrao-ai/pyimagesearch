# ------------------------
#   USAGE
# ------------------------
# python extract_embeddings.py --dataset dataset \
#	--embeddings output/embeddings.pickle \
#	--detector face_detection_model \
#	--embedding-model face_embedding_model/openface_nn4.small2.v1.t7

# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True, help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True, help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the serialized face detector from disk
print("[INFO] Loading the face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Load the serialized face embedding model from disk and set the preferable target
print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Grab the paths to the input images in the dataset
print("[INFO] Quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialize the list of extracted facial embeddings and the corresponding names
knownEmbeddings = []
knownNames = []

# Initialize the total number of processed faces
total = 0

# Loop over the image paths
for (i, imgPath) in imagePaths:
    # Extract the person name from the image path
    print("[INFO] Processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imgPath.split(os.path.sep)[-2]
    # Load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab
    # the image dimensions
    img = cv2.imread(imgPath)
    img = imutils.resize(img, width=600)
    (h, w) = img.shape[:2]
    # Construct the blob from the image
    imgBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # Apply the OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imgBlob)
    detections = detector.forward()
    # Ensure at least one face was found
    if len(detections) > 0:
        # We are making the assumption that each image has only ONE face, so find the bounding box
        # with the largest probability
        j = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, j, 2]
        # Ensure that the detection with the largest probability also means the minimum probability test (thus helping
        # filter out weak detection).
        if confidence > args['confidence']:
            # Compute the (x, y) coordinates of the bounding box for the face
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Extract the face ROI and grab the ROI dimensions
            face = img[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # Ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            # Construct a blob for the face ROI, then pass the blob through our face embedding model
            # to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # Add the name of the person + corresponding face embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1
# Dump the facial embeddings + names to the disk
print("[INFO] Serializing {} encodings...".format(total))
data = {"Embeddings": knownEmbeddings, "Names": knownNames}
f = open(args["Embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()
