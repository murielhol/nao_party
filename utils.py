import numpy as np
import os
import cv2
import json
from landmark_detection import landmark_detection


class LandmarkData:
    def __init__(self):
        self.image_dir_cali = "./calibration/"
        self.data_dir = "./final_run_data/complete_0_130"

        self.landmark_width = 0.1
        self.image_center = np.array([320, 240])
        self.image_range = 320 / np.tan(np.pi / 6)  # approximate to 30 degrees

        self._calibrate()

        file_seq = [i for i in range(130)]
        self.sensor_data = self._read_sensor_data(file_seq=file_seq,
                                                  data_dir="./final_run_data/complete_0_130",
                                                  prefix="recording")

    # find the distance of a landmark given the width (pixels)
    def _find_distance(self, pixel_width):
        return 1.0 * self.landmark_width * self.average_focal_length / pixel_width

    # find the bearing given the x offset from the center of image,
    # left is positive and right is negative
    def _find_bearing(self, landmark_position, image_seq):
        landmark_offset = self.image_center - landmark_position
        theta = np.arctan(landmark_offset[0] / self.image_range)

        robot_pos = self.sensor_data[image_seq]
        head_yaw = robot_pos['HeadYaw']
        theta = theta + head_yaw

        return theta

    def _calibrate(self):
        ''' use the calibration image to find the focal length '''

        focal_length = list()
        landmark_width_pix = list()
        detected_ind = list()

        actual_dist = [2, 1, 0.5, 3, 3, 2, 1, 0.5, 3, 2, 1, 0.5, 3, 2, 1, 0.5]

        for i in range(15):
            image_path = os.path.join(self.image_dir_cali, "calibration" + str(i) + ".jpg")
            image = cv2.imread(image_path)
            landmark_info = landmark_detection(image)

            if len(landmark_info) > 0:
                print landmark_info
                # focal length for every img
                focal_length.append(self._find_focal_length(landmark_info[2], actual_dist[i], self.landmark_width))

                # record the width of landmark in terms of pixels
                landmark_width_pix.append(landmark_info[2])

                # the index of landmarks that can be identified
                detected_ind.append(i)

        self.average_focal_length = np.array(focal_length).mean()
        print "Average focal length: {}".format(self.average_focal_length)

        estimated_dist = self.landmark_width * self.average_focal_length / np.array(landmark_width_pix)
        print "Estimated dist: {}".format(estimated_dist)

        actual_dist = np.array(actual_dist)
        print "Actual dist: {}".format(actual_dist[np.array(detected_ind)])

    @staticmethod
    def _find_focal_length(pixel_width, actual_dist, actual_width):
        return pixel_width * actual_dist / actual_width

    # read the data log into a list of dictionary, keys are 'HeadYaw', 'HeadPitch'...etc
    @staticmethod
    def _read_sensor_data(file_seq, data_dir, prefix):
        print 'reading sensor data log...'

        sensor_data = list()

        for i in file_seq:
            file_path = os.path.join(data_dir, prefix + str(i) + ".txt")

            with open(file_path, 'r') as f:

                data_entry = dict()
                for line in f:
                    line = line.rstrip().split(' ')

                    if len(line) == 2:
                        key, val = line[0], line[1]
                        data_entry[str(key)] = float(val)

                sensor_data.append(data_entry)

        return sensor_data

    # process all the images to get the info of all landmarks
    def _get_landmark_dataset(self, file_seq, outfile="./landmark_data.json"):
        """
         return a list of list of dict
         1st list: every element is an image
         2nd list: every element is a landmark
         dict: {signature: ID, range: R, bearing: radians}
         return: [ [ {ID:1, Range:2, Bearing:0.5}, {ID:3, Range:1.5, Bearing:0.2} ], [{ ... }] ]
        """

        detected_ind = list()
        data = list()

        for i in file_seq:
            image_path = os.path.join(self.data_dir, "recording" + str(i) + ".jpg")
            image = cv2.imread(image_path)
            landmark_info = landmark_detection(image)

            if len(landmark_info) > 0:
                print landmark_info

                landmarks = list()      # store the landmark information of a SINGLE image

                lm_width = landmark_info[2]
                for k, lm in enumerate(lm_width):     # assume the width is stored as a list

                    id = landmark_info[1][k]
                    distance = self._find_distance(lm)
                    bearing = self._find_bearing(landmark_info[3][k], i)  # use index i to find corresponding log

                    lm_info = {'id': id,
                               'distance': distance,
                               'bearing': bearing
                               }
                    landmarks.append(lm_info)   # append every landmark seen in an image

                data.append(landmarks)      # store all the landmarks in ONE image into the list
                detected_ind.append(i)    # the index of landmarks that can be identified

        with open(outfile, 'w') as f:
            json.dump(data, f)

        return data

