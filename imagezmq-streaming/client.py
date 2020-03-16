# -----------------------------
#   USAGE
# -----------------------------
# python client.py --server-ip SERVER_IP

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils.video import VideoStream
from imagezmq import ImageSender
import argparse
import socket
import time

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True, help="IP address of the server to which the client will connect")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# Initialize the ImageSender object with the socket address of the server
sender = ImageSender()

# Get the host name, initialize the video stream and allow the camera sensor to warmup
rpi_name = socket.gethostname()
print("[INFO] Camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

count = 0
while True:
	# Increment the counter and print its value to the console
	count = count + 1
	print("Sending: {}".format(count))
	# Read the frame from the camera and send it to the server
	frame = vs.read()
	sender.send_image(rpi_name, frame)
	time.sleep(1)
