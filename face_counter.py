import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
# 設定影像尺寸
width = 1440
height = 900

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while cap.isOpened():
    ret, frame = cap.read()
    # i = cv2.imread('1.jpg')
    print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    l = len(faces)
    print(l)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'face', ((w / 2).astype(int) + x,
                                    (y - h / 5).astype(int)),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    cv2.putText(frame, "face count", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2.0,
                (0, 255, 255), 2, 1)
    cv2.putText(frame, str(l), (230, 20), cv2.FONT_HERSHEY_PLAIN, 2.0,
                (0, 255, 255), 2, 1)
    #cv2.putText(i,"eyes count",(20,60),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2,1)
    print(
        frame.shape
    )  #cv2.putText(i,str(r),(230,60),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2,1)
    cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
