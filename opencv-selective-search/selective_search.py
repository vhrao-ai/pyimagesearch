# -----------------------------
#   USAGE
# -----------------------------
# python selective_search.py --image dog.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import random
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", type=str, default="fast", choices=["fast", "quality"], help="Selective search method")
args = vars(ap.parse_args())

# Load the input image
image = cv2.imread(args["image"])

# Initialize OpenCV's selective search implementation and set the input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# Check to see which selective search method is going to be used (in this case it's the fast but less accurate version)
if args["method"] == "fast":
    print("[INFO] Using *fast* selective search")
    ss.switchToSelectiveSearchFast()
# Otherwise, use the slower but more accurate version of selective search
else:
    print("[INFO] Using *quality* selective search")
    ss.switchToSelectiveSearchQuality()

# Run selective search on the input image
start = time.time()
rects = ss.process()
end = time.time()

# Show how along selective search took to run along with the total number of returned region proposals
print("[INFO] Selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))

# Loop over the regions proposals in chunks (in order to visualize them better)
for i in range(0, len(rects), 100):
    # Clone the original image in order to draw on it
    output = image.copy()
    # Loop over the current subset of region proposals
    for (x, y, w, h) in rects[i:i+100]:
        # Draw the proposal region for the bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
    # Show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(0) & 0xFF
    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

