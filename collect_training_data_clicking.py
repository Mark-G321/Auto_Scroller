
import face_recognition as fr
import numpy as np
import cv2 as cv
import pyautogui
import os
from pynput.mouse import Listener

root = r'C:\Users\marko\Downloads\CVT\testeyes3\\'
captured = {}


def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)


def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    # This function concatenates two images horizontaly
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv.hconcat(im_list_resize)

# This function finds the eyes in frame and returns an image with only the eyes


def findEyes(frame, left_eye, right_eye):
    # This function finds the eyes in frame and returns an image with only the eyes

    # get eye coordinates:
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


def scan(image_size=(64, 32)):
    _, frame = video_capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_features = fr.api.face_landmarks(frame)
    try:
        right_eye = face_features[0]['right_eye']
        left_eye = face_features[0]['left_eye']
        blank = np.zeros((480, 640), dtype="uint8")
        cv.fillPoly(blank, np.array([right_eye]), (255, 255, 255))
        cv.fillPoly(blank, np.array([left_eye]), (255, 255, 255))
        frame = cv.bitwise_and(frame, frame, mask=blank)
        eyes = findEyes(frame, left_eye, right_eye)
        eyes = cv.resize(eyes, (64, 32))
        eyes = normalize(eyes)
        return eyes
    except:
        return None


def on_click(x, y, button, pressed):
    # If the action was a mouse PRESS (not a RELEASE)
    if pressed:
        # Crop the eyes
        eyes = scan()
        # If the function returned None, something went wrong
        if not eyes is None:
            # Save the image
            counter = 0
            if (x, y) not in captured:
                captured[(x, y)] = 1
            else:
                counter = captured[(x, y)]
                captured[(x, y)] = captured[(x, y)]+1
            filename = root + "{}.{}.{}.jpeg".format(x, y, counter)
            cv.imwrite(filename, eyes)
            print("saved file", filename)


video_capture = cv.VideoCapture(0)
print("running")
with Listener(on_click=on_click) as listener:
    listener.join()
