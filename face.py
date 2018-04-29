import cv2
import numpy as np
# import serial

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
# 設定影像尺寸
width = 1440
height = 900

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# ser = serial.Serial('COM5', 115200)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    if len(faces):
        (x, y, w, h) = max(faces, key=lambda face: face[2]*face[3])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        print(type(gray))
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        position = x + w/2.0
        print(position)
        # if position < 320:
        #     ser.write(b'a')
        # else:
        #     ser.write(b'b')
        
    cv2.imshow('face', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()