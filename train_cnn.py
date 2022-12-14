import numpy as np
import time
import os
import cv2 as cv
from tensorflow import keras
import face_recognition as fr
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import pyautogui


root = r'C:\Users\marko\Downloads\CVT\testeyes3\\'
width, height = 1919, 1079
video_capture = cv.VideoCapture(0)

# normalizes the images in case they werent normaliezd during data gathering


def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)

# A cv2 function that concatantes two different sized images


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

# scans for a picture of the eyes


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

# loads the training images from the root directory


filepaths = os.listdir(root)
X, Y = [], []
for filepath in filepaths:
    _, y, _, _ = filepath.split('.')
    y = float(y)
    # no need to normalize here if the pictures are loaded are already normalized
    X.append(normalize(cv.imread(root + filepath)))
    if y > 550:
        Y.append(1)
    else:
        Y.append(0)
X = np.array(X) / 255.0
Y = np.array(Y)
#print(X.shape, Y.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 64, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(0.0001), loss="mean_squared_error")
# model.summary()

epochs = 30
for epoch in range(epochs):
    model.fit(X, Y, batch_size=16)

model.save('models/trial_model.h5')


counter = 0
while True:
    # gets eyes from webcam
    eyes = scan()
    if not eyes is None:
        eyes = np.expand_dims(eyes / 255.0, axis=0)
        # guesses whether to scroll or not s==1 for scroll
        s = model.predict(eyes)[0][0]
        s = round(s)
        print(s)  # for testing
        if s == 1:
            counter += 1
            # 15 chosen to ballance against random glances down
            # so user needs to hold gaze at the bottom of a page for a period before page gets scrolled
            if counter == 15:
                print("***SCROLLING DOWN***")
                pyautogui.scroll(-420)
                counter = 0
                # wait before counting again so user can adjust his gaze
                time.sleep(1)
