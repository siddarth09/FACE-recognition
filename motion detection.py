import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(0)

firstFrame = None
area = 1500
while True:

    _, img = cam.read()
    img = cv2.flip(img, 1)
    test = "normal"
    img = imutils.resize(img, width=500)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.GaussianBlur(grayimg, (21, 21), 0)
    if firstFrame is None:
        firstFrame = grayimg
        continue
    imgDiff = cv2.absdiff(firstFrame, grayimg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg,None,iterations=2)
    cnts=cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"
        print(text)
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)

    cv2.imshow("camerafeed", img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("e"):
        break

cam.release()
cv2.destroyAllWindows()
