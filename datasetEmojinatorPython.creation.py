# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:46:50 2019

@author: Yash
"""

import cv2
import numpy as np
import os
os.mkdir('gestures/')
image_x,image_y=50,50
def create_folder(folder_name):
    if not os.path.exists(folder_name):
       os.mkdir(folder_name)

def store_image(g_id):
   total_pics=1200
   cap=cv2.VideoCapture(0)#for single camera
   #image can on;y be taken in this area
   x,y,w,h=300,50,350,350
   
   create_folder("gestures/"+str(g_id))
   pic_no=0
   flag_start_capturing=False
   frames=0
   while True:
       
       ret,frame=cap.read()
       #all th eimages should be flipped
       frame=cv2.flip(frame,1)
       hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
       mask2=cv2.inRange(hsv,np.array([2,50,60]),np.array([25,150,255]))
       res=cv2.bitwise_and(frame,frame,mask=mask2)
       gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
       #In this, instead of box filter, gaussian
       #kernel is used. It is done with the function, cv2.GaussianBlur().
       #We should specify the width and height of kernel which should be positive and odd. We
       #also should specify the standard deviation in X and Y direction, 
       #sigmaX and sigmaY respectively. If only sigmaX is specified, 
       #sigmaY is taken as same as sigmaX. If both are given as zeros, they are calculated from kernel size. Gaussian blurring is highly effective in removing gaussian noise from the image.
       median=cv2.GaussianBlur(gray,(5,5),0)
       #returns a matrix of ones
       kernel_square=np.ones((5,5),np.uint8)
       #dilate will remove every other noise that is waste pixel outside of the 
       #image that is why we passed kernel_square of that size filled
       #with ones
       
       dilation=cv2.dilate(median,kernel_square,iterations=2)       
       #more smoother output
       opening=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel_square)
       #returns the image if the pixel value is >30 then 255 is assigned else 0
       ret,thresh=cv2.threshold(opening,30,255,cv2.THRESH_BINARY)
       thresh=thresh[y:y+h,x:x+w]
       #gives the outermost hierachial area
       contours=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
       
       if len(contours) > 0:
            
            contour = max(contours, key=cv2.contourArea)
            print(cv2.contourArea(contour))
            print(frames)
            if cv2.contourArea(contour)>10000 and frames>50:
               x1,y1,w1,h1=cv2.boundingRect(contour)
               pic_no+=1
               save_img=thresh[y1:y1+h1,x1:x1+w1]
               save_img=cv2.resize(save_img,(image_x,image_y))
               cv2.putText(frame,"Capturing...",(30,60),cv2.FONT_HERSHEY_TRIPLEX,2,(127,255,255))
               cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)
               print('Written')
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
       cv2.imshow("Capturing gesture", frame)
       cv2.imshow("thresh", thresh)
       keypress = cv2.waitKey(1)
       if keypress == ord('c'):
          if flag_start_capturing == False:
              flag_start_capturing = True
          else:
                flag_start_capturing = False
                frames = 0
       if flag_start_capturing == True:
            frames += 1
       if pic_no == total_pics:
            break
    
g_id = input("Enter gesture number: ")
store_image(g_id)
          
            
       
       
       
       
     