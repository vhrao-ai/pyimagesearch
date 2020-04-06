# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import cv2


# -------------------------------------------
#   FUNCTIONS
# -------------------------------------------
def anonymize_face_simple(image, factor=3.0):
    # Automatically determine the size of the blurring kernel based on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kH = int(h/factor)
    kW = int(w/factor)
    # Ensure the kernel width is odd
    if kW % 2 == 0:
        kW -= 1
    # Ensure the kernel height is odd
    if kH % 2 == 0:
        kH -= 1
    # Apply a Gaussian blur to the input image using the computed kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_face_pixel_rate(image, blocks=3):
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
            # Extract the ROI using a NumPy array slicing, compute the mean of the ROI, and then draw a rectangle
            # with the mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    # Return the pixelated blurred image
    return image
