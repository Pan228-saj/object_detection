import streamlit as st
import cv2
import numpy as np
from PIL import Image
from textblob import TextBlob
import pandas as pd






options=["Email","Mobile No"]
dataset=st.sidebar.selectbox("**Contact Me**",options)
if dataset=="Email":
    st.sidebar.text("""
Email-pankaj.sajwan20@gmail.com""")
else:
    st.sidebar.text("""8477979148""")

    
st.sidebar.title("About Me")
st.sidebar.text("""
Name-Pankaj Sajwan
B.Tech(Mechanical Engineering)                    
""")



st.title("Object Detection using webcam")
st.write("Detects faces using your webcam")
run=st.button('start webcam')




model=cv2.dnn_DetectionModel('C:/Users/pankaj/yolov4.cfg.txt','C:/Users/pankaj/yolov4.weights')
model.setInputSize(416,416)
model.setInputScale(1/255)

file=open("C:/Users/pankaj/objects_list.txt")
data=file.read()
file.close()
objects=data.split('\n')
vdo=cv2.VideoCapture(0)
faceModel=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    flag,img=vdo.read()
    if flag==False:
        break
    else:
        classes,probs,bboxs=model.detect(img,confThreshold=0.75,nmsThreshold=0.5)
        for box,cls,prob in zip(bboxs,classes,probs):
            x,y,w,h=box
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(img,f'{objects[cls]}',(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2)
            cv2.putText(img,str(round(prob,2)),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),1)
            cv2.imshow('object-detection',img)
        key=cv2.waitKey(20)
        if key==ord('c'):
             break

cv2.destroyAllWindows()
vdo.release()
