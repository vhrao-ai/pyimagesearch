# ------------------------
#   USAGE
# ------------------------
# python localize_text_tessract.py --image apple_support.png
# python localize_text_tessract.py --image apple_support.png --min-conf 50

# ------------------------
#   IMPORTS
# ------------------------
# Import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
ap.add_argument("-c", "--min-conf", type=int, default=0, help="mininum confidence value to filter weak text detection")
args = vars(ap.parse_args())

# Load the input image, convert it from BGR and RGB channel ordering,
# and use Tessract to localize each area of text in the input image
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

# Loop over each one of the individual text localizations
for i in range(0, len(results["text"])):
    # Extract the bounding box coordinates of the text region from the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # Extract the OCR text itself along with the confidence of the text localization
    text = results["text"][i]
    conf = int(results["conf"][i])
    # Filter out weak confidence text localizations
    if conf > args["min_conf"]:
        # Display the confidence and text in the terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")
        # Strip out non-ASCII text in order to draw the text on the image using OpenCV,
        # then draw a bounding box around the text along with the text itself
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# Show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
