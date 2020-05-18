# -----------------------------
#   USAGE
# -----------------------------
# python opencv_inpainting.py --image examples/example01.png --mask examples/mask01.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path input image on which we'll perform inpainting")
ap.add_argument("-m", "--mask", type=str, required=True, help="path input mask which corresponds to damaged areas")
ap.add_argument("-a", "--method", type=str, default="telea", choices=["telea", "ns"],
                help="inpainting algorithm to use")
ap.add_argument("-r", "--radius", type=int, default=3, help="inpainting radius")
args = vars(ap.parse_args())

# Initialize the inpainting algorithm to be Telea et al. method
flags = cv2.INPAINT_TELEA

# Check to see if Navier-Stokes (i.e, Bertalmio et al.) method is the right method for inpainting
if args["method"] == "ns":
    flags = cv2.INPAINT_NS

# Load the:
# (1) input image (i.e., the image we're going to perform inpainting on);
# (2) the  mask which should have the same input dimensions as the input image -- zero pixels correspond to areas
# that *will not* be inpainted while non-zero pixels correspond to "damaged" areas that inpainting will try to correct
image = cv2.imread(args["image"])
mask = cv2.imread(args["mask"])
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# Perform inpaiting using OpenCV
output = cv2.inpaint(image, mask, args["radius"], flags=flags)

# Show the original input image, mask and output image after applying the inpainting
cv2.imshow("Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Output", output)
cv2.waitKey(0)
