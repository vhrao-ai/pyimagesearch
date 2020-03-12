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
args = vars(ap.parse_args())

# Initialize the ImageSender object with the socket address of the server
sender = ImageSender(connect_to="tcp://{}:5555".format(args["server_ip"]))

# Get the host name, initialize the video stream and allow the camera sensor to warmup
rpi_name = socket.gethostname()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

while True:
	# Read the frame from the camera and send it to the server
	frame = vs.read()
	sender.send_image(rpi_name, frame)
