import numpy as np
import time
import cv2 as cv
from tensorflow import keras
import face_recognition as fr
from keras.models import *
import pyautogui

# normalizes the images


def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)

# A cv2 function that concatanates two images of differnt sizes


def hconcat_resize_min(im_list, interpolation=cv.INTER_CUBIC):
    # This function concatenates two images horizontaly
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv.hconcat(im_list_resize)


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

# captures a frame of the users eyes


def scan():
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


# loads a pre-trained model for demonstration
model = load_model('models/trial_model.h5')
video_capture = cv.VideoCapture(0)
counter = 0

while True:
    # get eyes
    eyes = scan()
    if not eyes is None:
        eyes = np.expand_dims(eyes / 255.0, axis=0)
        s = model.predict(eyes)[0][0]
        # s is the prediction of whether to scroll the screen or not
        s = round(s)
        print(s)  # for testing
        if s == 1:
            counter += 1
            # The counter helps against users glancing down, enforces that the gaze is held steadily beneath the threshold
            if counter == 15:
                print("***SCROLLING DOWN***")
                # scrolls down 420 pixels
                pyautogui.scroll(-420)
                counter = 0
                # wait timer so the user has time to adjust his gaze before counter starts counting again
                time.sleep(1)
