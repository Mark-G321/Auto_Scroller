
import face_recognition as fr
import numpy as np
import cv2 as cv
import pyautogui
import os


# the root directory of the project
root = r'C:\Users\marko\Downloads\CVT\testeyes3\\'

# normalizes the images
def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)

# A cv2 function that concatantes two pictures of different sizes
def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    # This function concatenates two images horizontaly
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv.hconcat(im_list_resize)

# Finds the eyes in frame and returns an image with only the eyes
def findEyes(frame, left_eye, right_eye):
    # This function finds the eyes in frame and returns an image with only the eyes

    # get eye coordinates from the list:
    # left eye
    min_x_left_eye = min(left_eye, key=lambda t: t[0])[0]
    max_x_left_eye = max(left_eye, key=lambda t: t[0])[0]
    min_y_left_eye = min(left_eye, key=lambda t: t[1])[1]
    max_y_left_eye = max(left_eye, key=lambda t: t[1])[1]
    # right eye
    min_x_right_eye = min(right_eye, key=lambda t: t[0])[0]
    max_x_right_eye = max(right_eye, key=lambda t: t[0])[0]
    min_y_right_eye = min(right_eye, key=lambda t: t[1])[1]
    max_y_right_eye = max(right_eye, key=lambda t: t[1])[1]

    # combine both eyes into a single image
    eyes = hconcat_resize_min([frame[min_y_left_eye:max_y_left_eye, min_x_left_eye:max_x_left_eye],
                               frame[min_y_right_eye:max_y_right_eye, min_x_right_eye:max_x_right_eye]])
    return eyes

# Takes @times * pictures of the user's eyes and saves them into @folder
def getEye(times=1, coords=(0, 0), folder="eyes", size=(300, 40), webcam=cv.VideoCapture(0)):

    os.makedirs(folder, exist_ok=True)
    for i in range(times):
        _, frame = webcam.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_features = fr.api.face_landmarks(frame)

        try:
            right_eye = face_features[0]['right_eye']
            left_eye = face_features[0]['left_eye']
            blank = np.zeros((480, 640), dtype="uint8")
            cv.fillPoly(blank, np.array([right_eye]), (255, 255, 255))
            cv.fillPoly(blank, np.array([left_eye]), (255, 255, 255))
            frame = cv.bitwise_and(frame, frame, mask=blank)
        except:
            print('Error at coord: ', coords, " please try again.")
            continue

        eyes = findEyes(frame, left_eye, right_eye)
        # To make sure frame is acceptable data

        eyes = cv.resize(eyes, size)
        eyes = normalize(eyes)

        cv.imwrite(
            folder + "/" + str(coords[0]) + "." + str(coords[1]) + "." + str(
                i) + ".jpg", eyes)
        blank = np.zeros((480, 640), dtype="uint8")

#scans the screen and takes an image of the user's eyes at every 100px location
webcam = cv.VideoCapture(0)
for i in range(100, 1900, 100):
    for j in range(100, 1000, 100):
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)

        getEye(times=10, coords=(i, j), folder="testeyes3",
               size=(64, 32), webcam=webcam)
