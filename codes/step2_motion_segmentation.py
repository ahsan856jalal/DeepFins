#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:59:19 2022

@author: ahsanjalal
"""

import numpy as np
import cv2
from pylab import *
from os.path import join, isfile
import sys,os,glob
from ctypes import *
import math
import random
def run_histogram_equalization(rgb_img):

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img
video_dir='Fish Data'
ms_save_dir='motion_segmentation' # saving directory
temp_dir='temp_dir'
os.makedirs(ms_save_dir,exist_ok=True)
os.makedirs(temp_dir,exist_ok=True)
a=open('videos_test.txt','r')
video_used=a.readlines()



# dim = (640, 640) 
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)) # opening kernel
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) # closing kernel
var_thresh=150 # threshold of variance in the pixels 
background_ratio=0.85 # sensitivity of the model to classify it as foreground, smaller the value more sensitive the model to noise
no_of_gmm=25 # number of Gaussian mixure models , higher value makes it more dynamic to background changes at the cost of more computation

# video_fols=sorted([f for f in os.listdir(video_dir) if not f.startswith('.')])
vid_counter=0

for vid_counter, vids in enumerate(video_used, 1):
#    a=input('press enter')
    video_name=vids.rstrip()
    v_split=video_name.split('.')
    frame_from_vid=int(v_split[2])
    frame_on_gt=frame_from_vid+60 
    if os.path.exists(join(ms_save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png')):
        print(f"Skipping {vid_counter}/{len(video_used)}: {video_name}")
        continue
    
    # if not os.path.exists(join(ms_save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png')):
    #     vid_counter+=1
    #     print(vid_counter)
        
    if not os.path.exists(join(ms_save_dir)):
        os.makedirs(join(ms_save_dir))  
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=var_thresh,detectShadows=False) # shadow false to avoid large fish contour
    fgbg.setBackgroundRatio(background_ratio) # set the minimum background ratio
    fgbg.setNMixtures(no_of_gmm) # setting Gaussian distributions
    cap = cv2.VideoCapture(join(video_dir,v_split[0],video_name))
    ret, frame = cap.read()
    if size(frame) !=1:
        frame=run_histogram_equalization(frame)
    # frame=cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    [img_h,img_w,ch]=shape(frame)
    counter=0
    while(ret):
       # frame=cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
       obj_arr=[]
       blobs=[]
       fgmask = fgbg.apply(frame,) # default settings where learning rate is automaticalyl selected
       
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1)
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
       
       # img_file="%03d.png" % counter
       if counter==60: 
           cv2.imwrite(join(ms_save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png'),fgmask)
           break
       ret, frame = cap.read()
       if size(frame) !=1:
           frame=run_histogram_equalization(frame)
        
       counter+=1
       k = cv2.waitKey(30) & 0xff
       if k == 27:
           break
    cap.release()
    # else:
    #     vid_counter+=1
    # break
    cv2.destroyAllWindows()