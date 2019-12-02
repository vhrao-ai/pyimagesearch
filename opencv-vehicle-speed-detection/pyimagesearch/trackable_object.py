# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np


# -----------------------------
#   Trackable Object
# -----------------------------
class TrackableObject:
    def __init__(self, object_id, centroid):
        # Store the object ID, then initialize a list of centroids using the current centroid
        self.object_id = object_id
        self.centroids = [centroid]
        # Initialize the dictionaries to store the timestamp and position of the object at various points
        self.timestamp = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.position = {"A": None, "B": None, "C": None, "D": None}
        self.lastPoint = False
        # Initialize the object speeds in MPH and KMPH
        self.speed_mph = None
        self.speed_kmph = None
        # Initialize two booleans, one to be used to indicate if the object's speed has already been estimated or not,
        # and another one to be used to indicate if the object's speed has been logged or not
        self.estimated = False
        self.logged = False
        # Initialize the direction of the object
        self.direction = None

    def calculate_speed(self, estimated_speeds):
        # Calculate the speed in KMPH and MPH
        self.speed_kmph = np.average(estimated_speeds)
        miles_per_one_kilometer = 0.621371
        self.speed_mph = self.speed_kmph * miles_per_one_kilometer
