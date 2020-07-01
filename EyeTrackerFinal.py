import cv2
import dlib
import imutils
import pyautogui
import os
import numpy as np
import LineOfBestFit

###################
# Author: Anthony Nguyen
# Date: ??? - 6/30/20
# Purpose:
#      Calculates where you are looking at on a screen via eye-tracking and machine learning
#
###################

def shapeToNP(shape):
    #makes numpy array of (x,y) coordinates for the 68 facial landmarks

    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def filtersForEye(threshold):
    #applying filters upon the input of the eye to make it easier to locate center of eye (even while moving)
    kernel = np.ones((9, 9), np.uint8)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    points = [shape[36:42]]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)

    mask = cv2.dilate(mask, kernel, 5)
    eyes = cv2.bitwise_and(frame, frame, mask=mask)
    mask = (eyes == [0, 0, 0]).all(axis=2)
    eyes[mask] = [255, 255, 255]
    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 3)
    img = cv2.bitwise_not(img)

    return img

def contouring(filteredInput, image):
    #takes the filtered input (for the eyes) and locates where the center of the eye is

    #finds the contours of the filtered input
    contours, _ = cv2.findContours(filteredInput, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        #calculating the moment of the filtered input to get coordinates of center

        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)
        return (cx, cy)
    except:

        #if it can't find any eyes, returns (-1, -1) so program can filter out these specific inputs
        return (-1, -1)

def getPictures(img, threshold, enableBorder, calibrationPoints):
    # "calibration sequence," where the program takes a picture of your eye while looking at dots on the screen.
    # intervals are divided into eighths, where 0-8 are on the horizontal axis and 9-17 are on vertical axis.

    #pulling up the calibration screen and making it full-screen
    calibration = cv2.imread("calibration.png")
    height, width, _ = calibration.shape

    cv2.namedWindow("calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    #creating the calibration dots

    #horizontal dots
    if calibrationPoints <= 8:

        #for dots 0 and 8, they need to be pushed inwards by 20 pixels, so they don't appear off-screen
        if calibrationPoints == 0:
            cv2.circle(calibration, (int(.125 * calibrationPoints * width + 20), 20), 8, (255, 0, 0), -1)

        elif calibrationPoints == 8:
            cv2.circle(calibration, (int(.125 * calibrationPoints * width - 20), 20), 8, (255, 0, 0), -1)

        else:
            cv2.circle(calibration, (int(.125 * calibrationPoints * width), 20), 8, (255, 0, 0), -1)

    #vertical dots
    elif calibrationPoints <= 17:

        #for dots 9 and 17, they need to be pushed inwards by 20 pixels, so they don't appear off-screen
        if calibrationPoints == 9:
            cv2.circle(calibration, (20, int(.125 * (calibrationPoints-9) * height + 20)), 8, (0, 0, 255), -1)

        elif calibrationPoints == 17:
            cv2.circle(calibration, (20, int(.125 * (calibrationPoints-9) * height - 20)), 8, (0, 0, 255), -1)

        else:
            cv2.circle(calibration, (20, int(.125 * (calibrationPoints-9) * height)), 8, (0, 0, 255), -1)

    #when the calibration sequence is over, close the calibration screen
    else:
        cv2.destroyWindow("calibration")
        return None

    #puts the threshold (for eye detection) and the status of enableBorder variable on the calibration screen
    cv2.putText(calibration, ("Threshold: " + str(threshold)), (int(.6 * width), int(.625 * height)),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.putText(calibration, ("Border Enabled: " + str(enableBorder)), (int(.6 * width), int(.75 * height)),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("calibration", calibration)

    key = cv2.waitKey(1)

    #whenever the 'c' key is typed, "writes" the input to a folder to be analyzed later
    if key == ord('c'):
        cv2.imwrite(("analysis\\pic" + str(calibrationPoints) + ".jpg"), eye)

        #returns True if an image has been written during an iteration of function
        # This allows the calibration and the eye-tracking process to occur simultaneously
        return True

    #returns False if calibration does not write an image file during an iteration of function
    return False

def analyzePictures():
    # since the eye-tracker notes the center of the eye to be red,
    # this program detects the red dots in the calibration pictures and saves them as coordinates for when eye is
    # looking at a certain spot (and triangulates from there)

    #stores the xPoints for the horizontal calibration and the yPoints for the vertical calibration
    xPoints = []
    yPoints = []

    lowerBound = np.array([0, 0, 254], dtype="uint8")
    upperBound = np.array([0, 0, 255], dtype="uint8")

    #iterates through the 18 images to find the red points
    for i in range(18):
        image = cv2.imread("analysis\\pic" + str(i) + ".jpg")

        #actually detecting all of the red pixels
        mask = cv2.inRange(image, lowerBound, upperBound)
        output = cv2.bitwise_and(image, image, mask=mask)
        points = cv2.findNonZero(mask)

        #calculating the average point for all the red pixels
        averagePoint = np.mean(points, axis=0)
        averagePoint = np.array(averagePoint[0])
        averagePoint = tuple(averagePoint)

        #stores the x-coordinate of images 0-8
        if 0 <= i and i < 9:
            xPoints.append([averagePoint[0]])

        #stores the y-coordinate of images 9-17
        elif 9 <= i and i < 18:
            yPoints.append([averagePoint[1]])

    #saves the arrays into numpy arrays and returns them
    xPointsNP = np.array(xPoints, dtype="double")
    yPointsNP = np.array(yPoints, dtype="double")

    return (xPointsNP, yPointsNP)

#sets up detector to detect facial landmarks (specifically, right eye)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#input from camera
video = cv2.VideoCapture(0)

#used for the "threshold" slider
cv2.namedWindow('eyes')
cv2.createTrackbar('threshold', 'eyes', 0, 255, lambda x : x)

#global variables
calibrationPoints = 0
calibrated = False
analyzed = False

#if True, program uses a static border for where it detects your eyes (instead of it moving)
enableBorder = False
border = (None, None, None, None)

#lambda functions used because they can adapt to different input parameters of linear/quadratic regression
lookingAtX = lambda x : x
lookingAtY = lambda x : x

#storing the regression line
horizonalRegressionLine = None
verticalRegressionLine = None

while (True):
    key = cv2.waitKey(1)

    _, frame = video.read()
    eye = frame.copy()

    #threshold slider
    threshold = cv2.getTrackbarPos('threshold', 'eyes')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shapeToNP(shape)

        mid = (shape[42][0] + shape[39][0])
        img = filtersForEye(threshold)

        #coordinates for center of eye
        (pupilX, pupilY) = contouring(img[:, 0:mid], frame)

        if enableBorder:
            #cannot enableBorder until it's gone through one iteration of loop to establish border variable

            eye = frame[border[1] - 10: border[1] + border[3] + 10, border[0] - 5: border[0] + border[2] + 5]
            eye = imutils.resize(eye, width=400, inter=cv2.INTER_CUBIC)

        else:

            (x, y, w, h) = cv2.boundingRect(np.array([shape[36:42]]))
            border = (x, y, w, h)

            eye = frame[y - 10: y + h + 10, x - 5: x + w + 5]
            eye = imutils.resize(eye, width=400, inter=cv2.INTER_CUBIC)

            #maybe when doing horizontal calibration, keep constant width,
            # and for vertical calibration, keep constant height?

            #if 0 <= calibrationPoints and calibrationPoints < 9 or calibrationPoints >= 18:
            #    eye = imutils.resize(eye, width=400, inter=cv2.INTER_CUBIC)
            #
            #elif 9 <= calibrationPoints and calibrationPoints < 18:
            #    eye = imutils.resize(eye, height=275, inter=cv2.INTER_CUBIC)

        cv2.imshow('right eye', eye)

    cv2.imshow('eyes', frame)

    if not calibrated:
        if calibrationPoints >= 18:
            calibrated = True

        #if picture taken, move to next calibration point
        if (getPictures(eye, threshold, enableBorder, calibrationPoints)):
            calibrationPoints += 1

    elif calibrated:
        if not analyzed:
            #numpy arrays of x-points for horizontal calibration and y-points for vertical calibration
            (xPoints, yPoints) = analyzePictures()

            #calibrations with both sets of points to see if linear/quadratic regression is more efficient
            xCalibration = LineOfBestFit.calibration(xPoints)
            yCalibration = LineOfBestFit.calibration(yPoints)

            #checking for each form of regression in calibration (and adapting variables to that)
            if (xCalibration[0] == "linear"):
                lookingAtX = lambda x: [[x]]
                horizonalRegressionLine = xCalibration[1]

            elif (xCalibration[0] == "polynomial"):
                lookingAtX = lambda x: xCalibration[2].fit_transform([[x]])
                horizonalRegressionLine = xCalibration[1]

            if (yCalibration[0] == "linear"):
                lookingAtY = lambda y: [[y]]
                verticalRegressionLine = yCalibration[1]

            elif (yCalibration[0] == "polynomial"):
                lookingAtX = lambda y: yCalibration[2].fit_transform([[y]])
                verticalRegressionLine = yCalibration[1]

            analyzed = True

        elif analyzed:

            #takes input from your screen and places a circle where the program calculates where you're looking at
            screen = pyautogui.screenshot()
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
            screen = imutils.resize(screen, width=900)

            height, width, _ = screen.shape

            #coordinates on screen of where you're looking at
            screenX = (horizonalRegressionLine.predict(lookingAtX(pupilX)) * width)[0][0]
            screenY = (verticalRegressionLine.predict(lookingAtY(pupilY)) * height)[0][0]

            #checks if the input is valid before placing circle
            # if (-1,-1), there's no point available
            # if something else, then it's impossible to place on screen bc it would be calculated as off-screen

            if (pupilX, pupilY) != (-1, -1) or (screenX < 0.0 and screenX > 1.0) or (screenY < 0.0 and screenY > 1.0):
                cv2.circle(screen, (int(screenX), int(screenY)), 2, (0, 255, 0), -1)

            #printing things to ensure smooth operation (and easy debugging)
            print("dimensions: " + str(height) + ", " + str(width))
            print("coords: " + str(screenX) + ", " + str(screenY))
            print("lookingAt: " + str(lookingAtX(pupilX)) + ", " + str(lookingAtY(pupilY)) + "\n")

            cv2.imshow("screen", screen)
            cv2.waitKey(1)

    #whenever 'w' key is pressed and there are 18 pictures in "analysis" folder (for calibration),
    # skips the calibration sequence
    if key == ord('w') and len(os.listdir("analysis")) == 18:
        calibrationPoints = 18

    #whenever 'e' key is pressed, toggles the "hard-lock" for border for bound of eye-tracking
    elif key == ord('e'):
        enableBorder = (not enableBorder)

    #whenever 'q' key is pressed, ends program
    elif key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()