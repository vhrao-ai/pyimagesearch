# -----------------------------
#   USAGE
# -----------------------------
# python ocr_non_english.py --image images/german.png --lang deu

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from textblob import TextBlob
import pytesseract
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image to be OCR'd")
ap.add_argument("-l", "--lang", required=True, help="Language that Tesseract will use when OCR'ing")
ap.add_argument("-t", "--to", type=str, default="en", help="Language that we'll be translating to")
ap.add_argument("-p", "--psm", type=int, default=13, help="Tesseract PSM mode")
args = vars(ap.parse_args())

# Load the input image and convert it from BGR to RGB channel ordering
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# OCR the image, supplying the country code as the language parameter
options = "-l {} --psm {}".format(args["lang"], args["psm"])
text = pytesseract.image_to_string(rgb, config=options)

# Show the original OCR text
print("ORIGINAL")
print("========")
print(text)
print("")

# Translate the text to a different language
tb = TextBlob(text)
translated = tb.translate(to=args["to"])

# Show the translated text
print("TRANSLATED")
print("==========")
print(translated)
