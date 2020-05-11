# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def detect_and_predict_age(image, faceNet, ageNet, minConf=0.5):
    # Define the list of age buckets for the age detector to predict and initialize the results list
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    results = []
    # Grab the dimensions of the image and construct a blob from the image
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e, probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > minConf:
            # Compute the (x, y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Extract the ROI of the face
            face = image[startY:endY, startX:endX]
            # Ensure the face ROI is sufficiently large
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            # Construct a blob from the image *just* the face ROI
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
            # Make predictions on the age and find the age bucket with the largest corresponding probability
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]
            # Construct a dictionary consisting of both the face bounding box location along with the age prediction,
            # then update the results list
            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
            }
            results.append(d)
    # Return the results to the calling function
    return results


def detect_camo(image, camoNet):
    # Initialize (1) the class labels the camo detector can predict and (2) the ImageNet means (in RGB order)
    CLASS_LABELS = ["camouflage_clothes", "normal_clothes"]
    MEANS = np.array([123.68, 116.779, 103.939], dtype="float32")
    # Resize the image to 224x224 (ignoring the aspect ratio), convert the image from BGR to RGB ordering, and then add
    # a batch dimension to the volume
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0).astype("float32")
    # Perform mean subtraction
    image -= MEANS
    # Make predictions on the input image and find the class label with the largest corresponding probability
    preds = camoNet.predict(image)[0]
    i = np.argmax(preds)
    # Return the class label and the corresponding probability
    return CLASS_LABELS[i], preds[i]


def anonymize_face_pixelate(image, blocks=3):
    # Divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # Loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # Compute the starting and ending (x, y) coordinates for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # Extract the ROI using NumPy array slicing, compute the mean of the ROI, and then draw a rectangle
            # with the mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    # Return the pixelated blurred image
    return image



