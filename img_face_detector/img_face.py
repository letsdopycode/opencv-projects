import cv2
import numpy as np


#haarcascade  frontal face classier
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\hp\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

#read image
img = cv2.imread("smile.jpg", 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#scale image in different size
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#draw rectangle on detected face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #show image
    cv2.imshow("frontface", img)
    if cv2.waitKey() == 0xff:
        break

cv2.destroyAllWindows()
cv2.destroyAllWindows()
