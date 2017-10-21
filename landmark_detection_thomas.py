import cv2
import numpy as np




def landmark_detection(image):

    '''
    returns signature, ID, size and location



    '''
    normalizedImg = np.zeros(np.shape(image))
    image = cv2.normalize(image,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    pink = []
    yellow = []
    green = []
    blue = []

    boundaries = [
        # pink
        ([145, 0, 0], [175, 255, 255]),
        # yellow
        ([20, 0, 0], [29, 255, 255]),
        # blue
        ([100, 0, 0], [150, 255, 255]),
        # green
        ([40, 0, 0], [90, 255, 255]),
    ]
    color_index = 0
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)

        output[np.where((output == [0,0,0]).all(axis=2))] = [255,255,255]
        if color_index == 1:
            output[np.where((output!=[255,255,255]).all(axis=2))] = [0,0,0]
        # blur
        kernel = np.ones((10,10), np.float32) / 100
        output = cv2.filter2D(output, -1, kernel)
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 220
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector(params)
        # Detect blobs.
        keypoints = detector.detect(output)
        # add to corresponding color
        if len(keypoints) > 0:
            for keypoint in keypoints:
                if color_index == 0:
                    pink.append(keypoint)
                elif color_index == 1:
                    yellow.append(keypoint)
                elif color_index == 2:
                    blue.append(keypoint)
                elif color_index == 3:
                    green.append(keypoint)
        
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(output, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show keypoints

        color_index += 1

    # % You also need the following information about the landmark positions: 
    # % cyan:magenta -1500 -1000 magenta:cyan -1500 1000 magenta:green 0 -1000 green:magenta 0 1000 yellow:magenta 1500 -1000 magenta:yellow 1500 1000 
    # % 0 -> green 1 -> magenta 2 -> yellow 3 -> blue  
    # L = [-15 -15 0 0 15 15;-10 10 -10 10 -10 10];
    # LID = [3 1 1 0 2 1;1 3 0 1 1 2];


    # print(pink)
    # print(yellow)
    # print(green)
    # print(blue)

    signatures = []
    for p in pink:
        size = p.size
        for y in yellow:
            # check if the keypoint is around the same size and if around the same x location
            if abs(p.pt[0]-y.pt[0]) < size/2 and y.size < size+(size/4) and y.size > size-(size/4):
                if p.pt[1] > y.pt[1]:
                    # pink top, yellow bottom
                    signatures.append([1, 2])
                    signatures.append([6])
                else:
                    # pink bottom, yellow top
                    signatures.append([2, 1])
                    signatures.append([5])
                signatures.append([size])
                signatures.append([p.pt])
        for b in blue:
            # check if the keypoint is around the same size and if around the same x location
            if abs(p.pt[0]-b.pt[0]) < size/2 and b.size < size+(size/4) and b.size > size-(size/4):
                if p.pt[1] > b.pt[1]:
                    # pink top, blue bottom
                    signatures.append([1, 3])
                    signatures.append([2])
                else:
                    # pink bottom, blue top
                    signatures.append([3, 1])
                    signatures.append([1])
                signatures.append([size])
                signatures.append([p.pt])
        for g in green:
            # check if the keypoint is around the same size and if around the same x location
            if abs(p.pt[0]-g.pt[0]) < size/2 and g.size < size+(size/4) and g.size > size-(size/4):
                if p.pt[1] > g.pt[1]:
                    # pink top, green bottom
                    signatures.append([1, 0])
                    signatures.append([3])
                else:
                    # pink bottom, green top
                    signatures.append([0, 1])
                    signatures.append([4])
                signatures.append([size])
                signatures.append([p.pt])

    return signatures






