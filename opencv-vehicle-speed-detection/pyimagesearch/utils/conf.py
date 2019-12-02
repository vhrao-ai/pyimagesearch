# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from json_minify import json_minify
import json


# -----------------------------
#   Configuration Class
# -----------------------------
class Conf:
    def __init__(self, conf_path):
        # Load and store the configuration and update the object's dictionary
        conf = json.loads(json_minify(open(conf_path).read()))
        self.__dict__.update()

    def __getitem__(self, key):
        # return the value associated with the supplied key
        return self.__dict__.get(key, None)
