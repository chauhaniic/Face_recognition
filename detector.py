
import cv2,os
import numpy as np
from PIL import Image 

path = os.path.dirname(os.path.abspath(__file__))
#recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer = cv2.face.LBPHFaceRecognizer_create() 
#recognizer = cv2.createLBPHFaceRecognizer()
recognizer.read(path+r'\trainer\trainer.yml')
#recognizer.load(path+r'\trainer\trainer.yml')
cascadePath = path+'\Classifiers\\face.xml'
#print(cascadePath)
#cascadePath = path+"\Classifiers\face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
#love='love'
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
#font = cv2.CV_FONT_HERSHEY_SIMPLEX #Creates a font
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #faces=faceCascade.detectMultiScale(gray, 1.3, 5)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        if(nbr_predicted==1):
             nbr_predicted='Baba'
        elif(nbr_predicted==2 or nbr_predicted==5):
             nbr_predicted='Pinki'
        elif(nbr_predicted==3):
             nbr_predicted='Sandy'
        elif(nbr_predicted==4):
             nbr_predicted='Lalu'
        elif(nbr_predicted==7):
             nbr_predicted='Apoorv'
        else:
            nbr_predicted='Unknown'
        cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 1.1, (0,255,0)) #Draw the text
        cv2.imshow('im',im)
        cv2.waitKey(10)









