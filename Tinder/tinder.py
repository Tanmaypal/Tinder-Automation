#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('c:\\users\\tanmay\\anaconda3\\envs\\tinder\\lib\\site-packages')
sys.path.append('c:\\users\\tanmay\\appdata\\local\\pip\\cache\\wheels\\a5\\82\\2c\\2d2ccc604e2c2e35994b89a173d922331f1c6e1af9320a7602')


# In[3]:


# !pip install keyboard


# In[4]:


import keyboard


# In[5]:


import dlib


# In[6]:


# !pip install opencv-python


# In[7]:


import tensorflow as tf
import numpy as np
import cv2
from PIL import ImageGrab
#import dlib
# import keyboard
import os
import pyautogui
import time
import cv2


# In[8]:


# !pip install import-ipynb


# In[9]:


import import_ipynb


# In[10]:


import helper


# In[11]:


from helper import is_tinder_open, press_dislike, press_like,pred


# In[12]:


THRESH=5
SCREEN_H,SCREEN_W=1920,1080
CAP_H,CAP_W=SCREEN_H//2,SCREEN_W//2
padding=50


# In[13]:


detector=dlib.get_frontal_face_detector()


# In[15]:


while(True):
    if is_tinder_open():

        screen_cap=ImageGrab.grab(bbox=(0,0,CAP_H,CAP_W))
        screen_cap_num=np.array(screen_cap)
        screen_cap=cv2.cvtColor(screen_cap_num,cv2.COLOR_BGR2RGB)
        screen_cap_copy=screen_cap.copy()
        faces=detector(screen_cap)

        if faces:

            face=faces[0]

            print(face)

            x,y,x1,y1=face.left(), face.top(), face.right() + padding,face.bottom()+padding

            cv2.rectangle(screen_cap,(x,y),(x1,y1),(0,255,0),2)

            face_img=screen_cap_copy[y:y1,x:x1]

            face_img=cv2.resize(face_img,(224,224))

            score=pred(face_img)

            if score> THRESH:
                press_like()
            elif score<=THRESH:
                press_dislike()
            cv2.imshow("face",face_img)
        else:
            press_dislike()
        cv2.imshow("screen_capture",screen_cap)
        if cv2.waitKey(1) &0xFF==ord('q'):
            break
    else:
        cv2.destroyWindow("Screen_capture")
        continue

cv2.destroyWindows()

