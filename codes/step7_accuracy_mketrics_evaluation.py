#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:38:53 2022

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



optical_dir='merged_SM_yolo'
gt_dir='original_test_frames'
rgb_dir='original_test_frames'
a=open('test_list.txt','r')
video_used=a.readlines()
# save_dir='/home/ahsanjalal/ozfish/_test_hist_results'
width=1920
height=1080


# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


###############################################################################
ttt=0
TP=0
FP=0
FN=0
total_gt=0
total_det=0    #cv2.imwrite(join(sot_classifier_dir,filename),rgb_copy)
MOTA=0
ct=0
iou_sum=0
vid_counter=0
# total_gt_count=0
for vids in video_used:
    vid_counter+=1
    print(vid_counter)
    video_name=vids.rstrip()
    video_name=video_name.split('/')[-1]
    v_split=video_name.split('.')
    frame_from_vid=int(v_split[2])
    # frame_on_gt=frame_from_vid+60
    frame_on_gt=frame_from_vid
    img_name=v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.txt'
    # frame=cv2.imread(join(rgb_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png'))
    if not os.path.exists(join(optical_dir,img_name)):
        yolo_txt=[]
    else:
        yolo_txt=open(join(optical_dir,img_name))##making it gt
        yolo_txt=yolo_txt.readlines()
        total_det+=len(yolo_txt)
        
    if not os.path.exists(join(gt_dir,img_name)):
        gt_txt=[]
    else:
        gt_txt=open(join(gt_dir,img_name))#making it yolo
        gt_lines=gt_txt.readlines()
        gt_count=len(gt_lines)
        total_gt+=gt_count
        ct=ct+gt_count
        # total_gt_count+=total_gt
    count_yolo=0 
    if len(yolo_txt)!=0: 
        for yolo_txt1 in yolo_txt:
         count_yolo+=1
         count_gt=0
         txt=yolo_txt1.rstrip()
         coords=txt.split(' ')
         label_gmm=int(coords[0])
         w_yolo=round(float(coords[3])*width)
         h_yolo=round(float(coords[4])*height)
         x_yolo=round(float(coords[1])*width)
         y_yolo=round(float(coords[2])*height)
         x_yolo=int(x_yolo)
         y_yolo=int(y_yolo)
         h_yolo=int(h_yolo)
         w_yolo=int(w_yolo)
         xmin_yolo = x_yolo - w_yolo/2
         ymin_yolo = y_yolo - h_yolo/2
         xmax_yolo = x_yolo + w_yolo/2
         ymax_yolo = y_yolo + h_yolo/2  
         xmin_yolo=int(xmin_yolo)
         xmax_yolo=int(xmax_yolo)
         ymin_yolo=int(ymin_yolo)
         ymax_yolo=int(ymax_yolo)
         if(xmin_yolo<0):
           xmin_yolo=0
         if(ymin_yolo<0):
           ymin_yolo=0
         if(xmax_yolo>width):
           xmax_yolo=width
         if(ymax_yolo>height):
           ymax_yolo=height
         match_flag=0
         count_gt_line=-1
         for line_gt in gt_lines:
           count_gt+=1  
           count_gt_line+=1  
           line_gt1 = line_gt.rstrip()
           coords=line_gt1.split(' ')
           label_gt=int(coords[0])
           
           w_gt=round(float(coords[3])*width)
           h_gt=round(float(coords[4])*height)
           x_gt=round(float(coords[1])*width)
           y_gt=round(float(coords[2])*height)
           x_gt=int(x_gt)
           y_gt=int(y_gt)
           h_gt=int(h_gt)
           w_gt=int(w_gt)
           xmin_gt = int(x_gt - w_gt/2)
           ymin_gt = int(y_gt - h_gt/2)
           xmax_gt = int(x_gt + w_gt/2)
           ymax_gt = int(y_gt + h_gt/2)
           if(xmin_gt<0):
               xmin_gt=0
           if(ymin_gt<0):
               ymin_gt=0
           if(xmax_gt>width):
               xmax_gt=width
           if(ymax_gt>height):
               ymax_gt=height
           # now calculating IOU 
           
           xa=max(xmin_yolo,xmin_gt)
           ya=max(ymin_yolo,ymin_gt)
           xb=min(xmax_yolo,xmax_gt)
           yb=min(ymax_yolo,ymax_gt)
           if(xb>xa and yb>ya):
               
               area_inter=(xb-xa+1)*(yb-ya+1)
               area_gt=(xmax_gt-xmin_gt+1)*(ymax_gt-ymin_gt+1)
               area_pred=(xmax_yolo-xmin_yolo+1)*(ymax_yolo-ymin_yolo+1)
               area_min=min(area_gt,area_pred)
               area_union=area_pred+area_gt-area_inter
               
           #now checking IOU area
               if(float(area_inter)/area_min>=0.5):
                   TP+=1
                   match_flag+=1
                   iou_sum+=float(area_inter)/area_min
                   # del gt_lines[count_gt_line]
                   break
                   
                   # if(label_yolo==label_gt):
                   #     TP+=1
                   #     num[label_gt] += 1
                   #     print('True count :gt_lines {}'.format(TP))
                   #     det_image+=1
               # else:
               #     FP+=1
               #     break # no break as instances where iou>50 due to overlap fish
               #     frame=cv2.rectangle(frame,(xmin_yolo,ymin_yolo),(xmax_yolo,ymax_yolo),(255,0,0),2)
               #     cv2.putText(frame,'FP',(int(x_yolo+2+w_yolo/2),int(y_yolo+h_yolo/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,0),3,cv2.LINE_AA)
               #     cv2.imwrite(join(save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png'), frame)

                   #     img_patch=img_rgb[ymin_yolo:ymax_yolo,xmin_yolo:xmax_yolo]
                   #     img_patch = cv2.resize(img_patch.astype('float32'), dsize=(50,50))
                   #     name1="%03d_%s" % (count_gt_line,img_file)
                   #     cv2.imwrite(join(save_main_dir_fp,name1), img_patch)
                   # del gt_lines[count_gt_line]


         if match_flag==0:
            FP+=1
            # frame=cv2.rectangle(frame,(xmin_yolo,ymin_yolo),(xmax_yolo,ymax_yolo),(255,0,0),2)
            # cv2.putText(frame,'FP',(int(x_yolo+2+w_yolo/2),int(y_yolo+h_yolo/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,0),3,cv2.LINE_AA)
            # cv2.imwrite(join(save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png'), frame)
            
            
           # img_patch=img_rgb[ymin_yolo:ymax_yolo,xmin_yolo:xmax_yolo]
           # img_patch = cv2.resize(img_patch.astype('float32'), dsize=(50,50))
           # if not os.path.exists(save_main_dir):
           #     os.makedirs(save_main_dir)
           # cv2.imwrite(save_main_dir+'/'+ "test_image.png", img_patch)
           # im = load_image(save_main_dir+'/'+ "test_image.png", 0, 0)
           # r = classify(net, meta, im)
           # r=r[0]
           # if r[0]=='background' or float(r[1])>0.8:
           #     # cv2.imwrite(save_main_dir+'/'+ r[0]+"_"+str(bkg_count)+"_.png", img_patch)
           #     # print('fish calss is {} and probability is {}'.format(r[0],float(r[1])))
           #     bkg_count+=1
           #     # print(bkg_count)
           # else:

           #     FP+=1
               # img_patch=img_rgb[ymin_yolo:ymax_yolo,xmin_yolo:xmax_yolo]
               # img_patch = cv2.resize(img_patch.astype('float32'), dsize=(50,50))
               # name1="%03d_%s" % (count_gt_line,img_file)
               # cv2.imwrite(join(save_dir_no_overlap,name1), img_patch)
                        # FP_no_overlap+=1
                        
                        
                        
                     
      
        
    else: # when both yolo and gmmOptical files are not present in respective folders
        FP+=gt_count
     
# print(num)
# print("Total GT detections are {}".format(total_gt))
# print("Total yolo_opt detections are {}".format(total_det))
FN=abs(total_gt-TP)      
# print('True positives are:  ', TP)
# print('False Positives are:   ', FP)
# print('False Neagatives are:   ', FN)
PR=float(TP)/(TP+FP) 
RE=float(TP)/(TP+FN)
print('DeepFins predictions are')
print (' Precision is :    ',PR)     
print (' Recall is :    ',RE )    
F_SCORE=float(2*PR*RE)/(PR+RE)
print (' F-score is :    ', F_SCORE)     
# print('MOTP score s   :   ',float(iou_sum)/ct)
