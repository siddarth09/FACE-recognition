import cv2,os
import numpy
alg="haarcascade_frontalface_default.xml"
face_detect=cv2.CascadeClassifier(alg)
datasets= 'datasets'
(images, labels, names , id)=([],[],{},0)
for (subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath +'/'+ filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1

(width , height)= (130, 100)
(images,labels)=[numpy.array(lis) for lis in [images,labels]]

model= cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)
print("TRAINING COMPLETED")


cam= cv2.VideoCapture(0)
cnt=0
while True:
    (_,img)= cam.read()
    img=cv2.flip(img,1)
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces= face_detect.detectMultiScale(grayImg,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        face=grayImg[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        prediction=model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))
        if prediction[1]<800:
            cv2.putText(img,'%s-%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-19),cv2.FONT_HERSHEY_PLAIN,4,(0,0,255))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(img,'Unkown',(x-10,y-19),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
            if(cnt>100):
                print("UNKOWN PERSON")
                cv2.imwrite("input.jpg",img)
                cnt=0
    cv2.imshow('OPENCV',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("e"):
        break

cam.release()
cv2.destroyAllWindows()

                  
