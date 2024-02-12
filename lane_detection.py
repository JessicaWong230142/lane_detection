import cv2
import numpy as np

#detects lane lines and midline by taking in frame of video
def detect_lines(frame):
    midpoint1 = None
    midpoint2 = None

    #finds height and width from frame
    height, width = frame.shape[:2]

    #calculates coordinates of rectangle mask using frame height and width
    mask_size = min(height, width) // 2
    mask_left = int((width - mask_size) / 2)
    mask_top = int((height - mask_size) / 2)
    mask_right = mask_left + mask_size
    mask_bottom = mask_top + mask_size

    #sets rectangle mask from frame
    mask = frame[mask_top:mask_bottom, mask_left:mask_right]

    #preprocesses frame with image processing: convert contents of rectangle mask to grayscale and finds edges in the mask
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    #detects lines from canny edge detector
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=400)

    #if lines are detected
    if lines is not None:
        #creates a list to store endpoints of detected lines
        line_endpoints = [[0] * 4 ] * len(lines)

        i = 0

        # adapted from: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html, date accessed: 2/7/24
        # also adapted from: https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/, date accessed: 2/8/24
        #loops over each detected line and sets its endpoints to match with coordinates in original frame
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 += mask_left
            x2 += mask_left
            y1 += mask_top
            y2 += mask_top

            #draws detected lines onto original frame and stores endpoints in the list line_endpoints
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            line_endpoints[i] = [x1, y1, x2, y2]
            i += 1

        counter = 0

        #checks if two or more lines are detected
        if len(line_endpoints) >= 2:
            #sets variables to store min x and y coords of the endpoints of first detected line
            min_x1 = line_endpoints[0][0]
            max_x1 = line_endpoints[0][0]
            min_y1 = line_endpoints[0][1]
            max_y1 = line_endpoints[0][1]

            #sets variables to store min x and y coords of the endpoints of second detected line
            min_x2 = line_endpoints[0][2]
            max_x2 = line_endpoints[0][2]
            min_y2 = line_endpoints[0][3]
            max_y2 = line_endpoints[0][3]

            #iterates over all detected lines to find min and max x and y coords for each of the two lines
            for k in range(len(line_endpoints)):
                min_x1 = min(min_x1, line_endpoints[k][0])
                max_x1 = max(max_x1, line_endpoints[k][0])
                min_y1 = min(min_y1, line_endpoints[k][1])
                max_y1 = max(max_y1, line_endpoints[k][1])

                min_x2 = min(min_x2, line_endpoints[k][2])
                max_x2 = max(max_x2, line_endpoints[k][2])
                min_y2 = min(min_y2, line_endpoints[k][3])
                max_y2 = max(max_y2, line_endpoints[k][3])

            #calculates midppoint of the two lines and draws a line connecting them (centerline)
            midpoint1 = ((min_x1 + max_x1) //2,
                         (min_y1 + max_y1) //2)
            midpoint2 = ((min_x2 + max_x2) //2,
                         (min_y2 + max_y2) //2)
            cv2.line(frame, midpoint1, midpoint2, (255, 255, 0), 3)
        #only draws avg line if less than two detected lines
        else:
            if counter < 11 and midpoint1 is not None and midpoint2 is not None:
                cv2.line(frame, midpoint1, midpoint2, (255, 255, 0), 3)
                counter += 1

    #draws rectangle around rectangle mask
    cv2.rectangle(frame, (mask_left, mask_top), (mask_right, mask_bottom), (255, 255, 255), 2)

    #returns frame with lines and rectangle mask drawn
    return frame
