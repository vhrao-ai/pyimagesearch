# ------------------------
#   IMPORTS
# ------------------------
import random
import os
import subprocess
import sys


# ------------------------
#   SPLIT DATASET
# ------------------------
def split_data_set(img_dir):
    f_val = open("snowman_test.txt", 'w')
    f_train = open("snowman_train.txt", 'w')
    path, dirs, files = next(os.walk(img_dir))
    data_size = len(files)
    ind = 0
    data_test_size = int(0.1 * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)
    for f in os.listdir(img_dir):
        if f.split(".")[1] == "jpg":
            ind += 1
            if ind in test_array:
                f_val.write(img_dir + '/' + f + '\n')
            else:
                f_train.write(img_dir + '/' + f + '\n')

