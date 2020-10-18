import cv2
import os

haar_file = "haarcascade_frontalface_default.xml"
datasets = "datasets"
sub_data = 'sujatha'

path = os.path.join(datasets, sub_data)

if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

count = 1

while count < 31:
    print(count)
    (_, img) = cam.read()
    img=cv2.flip(img,1)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = grayImg[y: y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s%s.png' % (path, count), face_resize)
    count += 1
    cv2.imshow('OPENCV', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("e"):
        break

cam.release()
cv2.destroyAllWindows()

