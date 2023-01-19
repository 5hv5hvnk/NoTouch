from django.shortcuts import render
import pyrebase
from django.http import HttpResponse
import requests
import os
import cv2
import numpy as np
import copy
import pandas as pd
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText
from email import encoders
from home.finegerprint_pipline import fingerprint_pipline

from tensorflow import keras
from PIL import Image
import glob


from keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Create your views here.

config={

#enter database configs
}

#configure firebase and download images

firebase=pyrebase.initialize_app(config)
storage = firebase.storage()
firebase_api="link"
url=storage.get_url(firebase_api)

response=requests.get(url)
data=response.json()

for imgbucket in data['items']:
    path_on_cloud=imgbucket['name']
    dir_name=path_on_cloud.split('/')[0]
    img_name=path_on_cloud.split('/')[1]
    suffix=".jpeg"
    if (img_name.endswith(suffix)):
        img_name=img_name
        # +".jpeg"
    else:
        img_name=img_name+".jpeg"
        # +".jpeg"
    dir_on_local="D:/capstone_django/notouch/static/images"

    if dir_name == "Myimages":
        dir_test_train="app upload"
    else:
        dir_test_train="original dataset"

    path_on_local=os.path.join(dir_on_local,dir_test_train,img_name)
    storage.child(path_on_cloud).download(path_on_local,path_on_local)

#segmenting and predicting
dir_on_local+='/'
train_folder_path=os.path.join(dir_on_local,"app upload")
test_folder_path=os.path.join(dir_on_local,"original dataset")

def extractSkin(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)


    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def cosine(A,B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))


def pred_on_img(img1,img2):

    img1 = cv2.resize(img1,dsize=(450, 700), interpolation=cv2.INTER_CUBIC)
    img1 = cv2.merge((img1,img1,img1))
    img2 = cv2.resize(img2,dsize=(450, 700), interpolation=cv2.INTER_CUBIC)
    
    cnn = tf.keras.models.Sequential([
    layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(700, 450,3)), #layer 1
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'), #layer 2
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'), #layer 3
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(50, activation='relu'), ##reducing nodes
    layers.Dense(30, activation='softmax'), ])
    #75% accu:- 4 layers 134 each with dropouts 0.25 and relu lr = 0.001 ann = 50,30,1
    opt = keras.optimizers.Adam(learning_rate=0.001)
    cnn.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    cnn.load_weights("pretrained_weights.h5")
    x_test=np.array([img1,img2])
    
    y_pred = cnn.predict(x_test)
    if cosine(y_pred[0],y_pred[1])>=0.9:
        #print(cosine(y_pred[0],y_pred[1]))
        return 1
    else:
        #print(cosine(y_pred[0],y_pred[1]))
        return 0

def task(path):
    img = extractSkin(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    min_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    #cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    max_img = cv2.convertScaleAbs(max_img)
    min_img = cv2.convertScaleAbs(min_img)

    # create a binary mask with the local maximum and minimum values
    mask = np.zeros_like(img)
    mask[max_img > min_img] = 255

    # apply the mask to the original image to create a binary image
    binary = cv2.bitwise_and(img, mask)
    return binary


def attendance():

    img_list=os.listdir(train_folder_path)
    img_list2=os.listdir(test_folder_path)
    roll=[]
    resu=[]
    for student_img in img_list:
        if student_img not in img_list2:
            continue
        student_img_path=train_folder_path+'/'+student_img
        original_student_img_path=test_folder_path+'/'+student_img

        skin=task(student_img_path)

        grayImage=skin
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

        thin_image=fingerprint_pipline(blackAndWhiteImage)
        # cv2.imshow('thin',thin_image)
        
        
        img2=cv2.imread(original_student_img_path)

        result=pred_on_img(thin_image,img2)

        roll_no=student_img.split('.')[0]

        # dict={'Roll Number':roll_no,'Present(1)/Absent(0)':result}
        # res.append(dict,ignore_index=True)
        # print(dict)
        # print(res)

        roll.append(roll_no)
        

    
    res=pd.DataFrame(list(zip(roll, resu)),columns=['Roll Number', 'Present(1)/Absent(0)'])
    # print(res)
    res.to_csv("D:/capstone_django/notouch/home/attendance.csv")


attendance()

#sending email

From = "sender email"
To = "reciever email"

message = MIMEMultipart()

message['From'] = From
message['To'] = To
message['Subject'] = "Attendance List"
body_email = "Sir/Ma'am \nKindly find the attendance list in the Excel file attached below \nThanks and regards \nTeam NoTouch"

message.attach(MIMEText(body_email, 'plain'))

filename = "attendance.csv"
attachment = open("D:/capstone_django/notouch/home/"+filename, "rb")

x = MIMEBase('application', 'octet-stream')
x.set_payload((attachment).read())
encoders.encode_base64(x)

x.add_header('Content-Disposition', "attachment; filename= %s" %filename)
message.attach(x)

s_e = smtplib.SMTP("smtp.gmail.com:587")
s_e.starttls()

s_e.login(From, "your 2 authentication factor key")
text = message.as_string()
s_e.sendmail(From, To, text)
s_e.quit()



def index(request):
    return HttpResponse("NoTouch Running")