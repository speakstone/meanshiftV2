# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 13:50:15 2018

@author: lilei0129
"""
from PIL import Image  
import numpy as np
import scipy
import matplotlib  
import cv2
import os 
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize  

####像素处理   
def change_Bi(image_name):
    change_Matrix = cv2.imread(image_name)
     ##更改只读模式
    change_Matrix.flags.writeable = True  
    
    ##获取长宽
    image_Matrix = change_Matrix[:,:]
    image_Row = (image_Matrix.shape)[0]
    image_Column = (image_Matrix.shape)[1]

    ##改变颜色为黑白二值图,过滤不需要的前景
    for Row in range(image_Row):
        for Column in range(image_Column):
            if 150<image_Matrix[Row,Column,0] < 210 or 100<image_Matrix[Row,Column,0] <200 or 100<image_Matrix[Row,Column,0] <200 :
                image_Matrix[Row,Column,0] =image_Matrix[Row,Column,1]=image_Matrix[Row,Column,2] =255
            else:
                image_Matrix[Row,Column,0] =image_Matrix[Row,Column,1]=image_Matrix[Row,Column,2] =0
            
    ##增加边缘
   
#    for j in range(7,image_Row-7):
#        i = 7
#        for i in range(7,image_Row-7):
#            image_edge = image_Matrix[i-1:i+2,j-1:j+2,0]
#            image_edge_sum = np.sum(image_edge)
#            print image_edge_sum
#            if 4*255 < image_edge_sum < 9*255:
#                image_Matrix[i,j,0] =image_Matrix[i,j,1]=image_Matrix[i,j,2] =100
#                i +=7
#                if i > image_Row-7:
#                    break
##                image_Matrix[i+1,j,0] =image_Matrix[i+1,j,1]=image_Matrix[i+1,j,2] =255
##                image_Matrix[i+2,j,0] =image_Matrix[i+2,j,1]=image_Matrix[i+2,j,2] =255


    Grayimg = cv2.cvtColor(image_Matrix, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(Grayimg, 12, 255,cv2.THRESH_BINARY)  
     
#    outputName = IMAGE_PRINT + 'BI_' + print_name
    ###图像加黑色边框，防止泊松融合溢出    
    cv2.rectangle(image_Matrix,(0,0),(image_Column-1,image_Row-1),(0,0,0),5)
    ##g改变尺寸
#    image_Matrix=cv2.resize(image_Matrix,(width,height),interpolation=cv2.INTER_CUBIC)  
#    cv2.imwrite(outputName, image_Matrix)
#    print image_Row
 

IMAGE_DIR = '/home/lilei/pbas_cnn/lady_image/lady_image_print/'
IMAGE_PRINT = '/home/lilei/pbas_cnn/lady_image/lady_image_BI/'

#for file in os.listdir(IMAGE_DIR):
#    if (file[-3:] != "jpg" and file[-3:] != "png"): continue
#    if (file[-9:] != "color.jpg" and file[-3:] != "color.png"): continue
#    name = os.path.join(IMAGE_DIR,file)
#    change_Bi(name,file)

name = os.path.join(IMAGE_DIR,"0_seg_color.jpg")
name2  = "0_seg_color.jpg"
change_Bi(name,name2)
