import cv2
import numpy as np
#from scipy import signal
from matplotlib import pyplot as plt
import imutils
#from sort import *


#mot_tracker = Sort() 
cap= cv2.VideoCapture('Mech Block PTZ 125 - P5414_camera_2019-02-06_09h00m30s_000.MOV')
count=0
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    ret, frame3 = cap.read()
    ret, frame4 = cap.read()
    ret, frame5 = cap.read()
    ret, frame6 = cap.read()
    fgmask = fgbg.apply(frame1)
    kernel = np.ones((3,3),np.uint8)
    opening =  cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    median = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    ret,th=cv2.threshold(median,20,295,cv2.THRESH_BINARY)
    dilated=cv2.dilate(th,np.ones((2,2),np.uint8),iterations=3)
    #dilated = cv2.Canny(dilated,50,50,3)
    #cv2.imshow('frame',fgmask)
    #i=cv2.imread('frame0.jpg')

    
    #im2=cv2.absdiff(frame3,frame1) 
    #gray = cv2.cvtColor(fgmask,cv2.COLOR_BGR2GRAY)
    #blur =cv2.GaussianBlur(fgmask,(3,3),0)
    #ret,th=cv2.threshold(blur,20,295,cv2.THRESH_BINARY)
    #dilated=cv2.dilate(th,np.ones((1,1),np.uint8),iterations=3)
    #edges = cv2.Canny(dilated,50,50,3)
    #img,c,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    
    template =cv2.imread('NDCD_cropped.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #template_gray = cv2.Canny(template_gray,50,50,3)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(dilated,template_gray,cv2.TM_CCOEFF_NORMED)
    (minval,maxval, minloc, maxloc) = cv2.minMaxLoc(res)
    #maxval=res.all()
    threshold = 0.41
    loc = np.where( res >= threshold)
    ocl=[]
    tol=[]
    cox=[]
    coy=[]
    col=(zip(*loc[::-1]))
    #print(col)
    for ocl in  (col):
      tol.append(ocl)
    #print(tol)    
    
    for i in range(0,len(tol)-1):
        if(tol[i+1][1]>=15+tol[i][1]):
                       
            cox.append(tol[i][0])
            cox.append(tol[i+1][0])          
            coy.append(tol[i][1])
            coy.append(tol[i+1][1])
        if(tol[i+1][0]>=15+tol[i][0]):
          if(tol[i+1][0]<=15+tol[i][0]): 
                cox.append(tol[i][0])
                cox.append(tol[i+1][0])          
                coy.append(tol[i][1])
                coy.append(tol[i+1][1])
             
            
                                
    #print(cox)
    #print(coy)
    #print(*loc[::-1]
    #top=maxval'
    cop=[]
    #print(loc)
    print(cox)
    print(coy)
    if (len(cox)==0):
        if(len(tol)!=0):
          cox.append(tol[0][0])          
          coy.append(tol[0][1])
          cv2.rectangle(frame1, (cox[0],coy[0]), (cox[0] + w, coy[0] + h), (0,0,255),2)
    i2=[]
    #track_bbs_ids = mot_tracker.update()
    #for top in zip(cox,coy):
    
    
    #cv2.putText(frame1,'Human', (top[0]- 10, top[1] - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
    #img_crop=frame1[top[1]:top[1] + h,top[0]:top[0] + w]
    #cv2.imshow('orig',img_crop)
    #cv2.imshow("g",fgmask)
      #cv2.imwrite("me%d.jpg" % count, img_crop)
      
      #plt.show()
      
      
     
      #threshold=170
      #val=[]

      #for i in range (0,len(a)):
       #if (a[i]>threshold):

        #c =print('1')
        #val.append(c)
       #else:
        #c=print('0')
        #val.append(c)
    #np.savetxt('arush-data.txt', normalized, fmt="%f")
    #cop.append(top)  
    #print("bitch")
    
    #cv2.imwrite("m%d.jpg" % count, frame1)
    cv2.imshow('original',frame1)
    
    #cv2.imshow('orig',frame1)
    #cv2.imshow('Detected',dilated)
    
    #plt.imshow(frame1)
    #plt.show
    #print(top)
       
    
      #cv2.imshow('ROI',edge)
      #cv2.imwrite("frame%d.jpg" % count, im) 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    count+=1
   


cap.release()
cv2.destroyAllWindows()
