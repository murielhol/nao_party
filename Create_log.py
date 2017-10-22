import numpy as np
import os
import cv2
import json
from landmark_detection import landmark_detection
from utils import *
import math
import matplotlib.pyplot as plt
import pickle

# landmark_data = LandmarkData()
# signatures = landmark_data._get_landmark_dataset([i for i in range(198)], outfile="./landmark_data.json")
signatures = pickle.load( open( "save.p", "rb" ) )

print 'reading location data log...'

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

location = list()
motion = list()
data_dir = "./data/"
prefix = "recording"
x_old = 0
y_old = 0
theta_old = 0
top = [3, 1, 1, 0, 2, 1]
bottom = [1, 3, 0, 1, 1, 2]

for i in range(198):
    file_path = os.path.join(data_dir, prefix + str(i) + "abs_loc.txt")
    with open(file_path, 'r') as f:

        data_entry = dict()
        for line in f:
            line = line.rstrip()
            if line[0] == "[" and line[-1] == "]":
                line = line[1:-1]
            line = line.split(',')
        if i == 0:
            x_0 = float(line[0])
            y_0 = float(line[1])
            theta_0 = float(line[2])
            x = 0
            y = 0
            theta = 0
        else:
            x = float(line[0]) - x_0
            y = float(line[1]) - y_0
            theta = float(line[2]) - theta_0

    location.append([x,y,theta])
    motion.append([math.sqrt((x-x_old)**2+(y-y_old)**2),theta-theta_old])
    x_old = x
    y_old = y
    theta_old = theta


location_array = np.array(location)
plt.plot(np.transpose(location_array)[0],np.transpose(location_array)[1],'-o')
plt.show()

file = open("log_new.txt", "w")
file.write("pose \t control  \t N-landmarkslandmarks: \t top \t bottom \t range \t bearing \n")

for i in range(198):
    head_Yaw = signatures[i][0]['head_yaw']
    print(signatures[i])
    landmarks = []
    for landmark in signatures[i][1:]:
        landmarks.append([len(signatures[i][1:]), bottom[landmark['id']-1], top[landmark['id']-1], landmark['distance'], (landmark['bearing']+head_Yaw)])

    writeline = flatten([location[i],motion[i], landmarks])
    for elem in writeline:
        elem = "%.6f " % float(elem)
        file.write(elem)
    file.write("\n")

file.close()



