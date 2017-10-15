import sys
import numpy as np
import cv2
import time
from naoqi import ALProxy
import random

if (len(sys.argv) <= 1):
    print "parameter error"
    print "python " + sys.argv[0] + " <ipaddr>"
    sys.exit()
ip_addr = sys.argv[1]
# port_num = int(sys.argv[2])# get NAOqi module proxy
port_num = 9559
videoDevice = ALProxy('ALVideoDevice', ip_addr, port_num)  # subscribe top camera
AL_kTopCamera = 1
AL_kVGA = 2  # 320x240
AL_kBGRColorSpace = 13
AL_kCameraExposureID = 20
captureDevice = videoDevice.subscribeCamera("test", AL_kTopCamera, AL_kVGA, AL_kBGRColorSpace, 10)  # create image

motionProxy = ALProxy("ALMotion", ip_addr, 9559)

width = 640
height = 480
image = np.zeros((height, width, 3), np.uint8)
num = 200

# turn off exposition(0-1)
expositionID = 11
videoDevice.setCameraParameter(captureDevice, expositionID, 0)
# set exposure time(0-255)
amountMS = 55
exposureID = 17
videoDevice.setCameraParameter(captureDevice, exposureID, amountMS)
# set gain(32-255)
amountGain = 255
gainID = 6
videoDevice.setCameraParameter(captureDevice, gainID, amountGain)
starttime = time.time()

while True:
    # get image and jointvalues
    jointvalues = motionProxy.getAngles("Body", True)
    jointnames = motionProxy.getBodyNames("Body")
    print(jointvalues)
    print jointnames
    result = videoDevice.getImageRemote(captureDevice);

    if result == None:
        print 'cannot capture.'
    elif result[6] == None:
        print 'no image data string.'
    else:  # translate value to mat
        values = map(ord, list(result[6]))
        image = np.reshape(values, (height, width, 3)).astype('uint8')

        cv2.imshow("pepper-top-camera-320x240", image)
        if (cv2.waitKey(5) == 97):  # 97 = a. press a to make a picture.
            # save joint values
            string2 = "./images/plaatje" + str(num) + ".txt"
            file = open(string2, "w")

            for jointvalue, jointname in zip(jointvalues, jointnames):
                file.write(jointname + " " + str(jointvalue) + "\n")
            file.close()

            # save image
            string2 = "./images/plaatje" + str(num) + ".jpg"
            cv2.imwrite(string2, image)
            num += 1

    # exit by [ESC]
    if cv2.waitKey(5) == 27:
        break

        # moving head every 0.5 sec
        # if(time.time() - starttime > 0.5):
        #   starttime = time.time()
        #   motionProxy.setAngles("HeadPitch", random.uniform(-0.35, 0.41), 0.6)

# set values to default to be kind to other people
hans = videoDevice.setAllCameraParametersToDefault(captureDevice)