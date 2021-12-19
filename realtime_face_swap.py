# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:37:28 2021

@author: ACER
"""

import cv2

stream1 = cv2.VideoCapture(0)
stream2 = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while 1:
    ret, frame1 = stream1.read()
    ret, frame2 = stream2.read()

    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces_f1 = face_cascade.detectMultiScale(gray, 1.4, 4)

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    faces_f2 = face_cascade.detectMultiScale(gray, 1.4, 4)

    for x, y, w, h in faces_f1:
        pass
        #cv2.rectangle(frame1, (x, y), (x+w, y+h), (255,0,0), 2)
        #frame1[y:y+h, x:x+w] = cv2.resize(frame2[y1:y1+h1, x1:x1+w1],(w, h))

    for x1, y1, w1, h1 in faces_f2:
        z = frame2.copy()
        #cv2.rectangle(frame2, (x1, y1), (x1+w1, y1+h1), (255,0,0), 2)
        frame2[y1:y1+h1, x1:x1+w1] =  cv2.resize(frame1[y:y+h, x:x+w],(w1, h1))
        frame1[y:y+h, x:x+w] = cv2.resize(z[y1:y1+h1, x1:x1+w1],(w, h))

    video1_h1 = int(stream1.get(4)) # height
    video1_w1 = int(stream1.get(3)) # width

    frame2 = cv2.resize(frame2,(video1_w1, video1_h1), interpolation = cv2.INTER_AREA)
    frame_combined = cv2.resize(frame1,(video1_w1*2, video1_h1), interpolation = cv2.INTER_AREA)
    frame_combined[:video1_h1,:video1_w1] = frame1
    frame_combined[:video1_h1, video1_w1:video1_w1*2] = frame2

    cv2.imshow('Joint Stream', frame_combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream1.release()
stream2.release()
cv2.destroyAllWindows()
