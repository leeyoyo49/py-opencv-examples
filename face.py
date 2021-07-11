import tkinter
import cv2
import numpy as np
import time, os
import tensorflow.keras
from PIL import ImageOps
import PIL
import numpy as np
from tkinter import *
import tkinter.messagebox

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

def use_keras():
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = PIL.Image.open("test.jpg")

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, PIL.Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return prediction

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# 設定影像尺寸
width = 1440
height = 900

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# ser = serial.Serial('COM5', 115200)

while True:
    try:
        os.remove("test.jpg")
    except:
        pass
    for x in range(100):
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 2, 5)
        if len(faces):
            (x, y, w, h) = max(faces, key=lambda face: face[2]*face[3])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv2.imwrite("test.jpg",roi_color)
            position = x + w/2.0

        cv2.imshow('face', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        time.sleep(0.03)
    try:
        b = use_keras()
        others = int(b[0,0]*1)
        big = int(b[0,1]*100)
        girl =int(b[0,2]*40) 
        fish = int(b[0,3]*5)
        print(others, big, girl, fish)
        total = int(others+big+girl+fish)
        tkinter.messagebox.showinfo("甲長 保正",f'大牛和你的契合度:{total}%')
        print(b)
    except:
        pass
