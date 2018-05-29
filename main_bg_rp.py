# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:12:53 2018

@author: lilei0129

预先使用deeplab等网络生成segment载入
"""

import numpy as np
import scipy
import math
from scipy.sparse import linalg
from scipy.misc import toimage
from scipy.misc import imshow
import cv2
from PIL import Image
import time
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize  
import blendingPoisson_alpha 
import blendingPoisson_bianyuan
import os 

start = time.time()    
##生成视频
foreImageName = 'lady_image/lady_image/' +str(1)+'.jpg'  
foreImg = cv2.imread(foreImageName,cv2.IMREAD_UNCHANGED) / 255.0      
rows = foreImg.shape[1] 
cols = foreImg.shape[0]
#fps=25
#fourcc=VideoWriter_fourcc(*"MJPG")  
#size = (rows,cols)
#videoWriter=cv2.VideoWriter('lady_image/test5.avi', fourcc, fps, size)      

def change_size(fIMG,bIMG):
    fx = float(fIMG.shape[0])
    fy = float(fIMG.shape[1])
    bx = float(bIMG.shape[0])
    by = float(bIMG.shape[1])
    ff = fx/fy
    byc = bx/ff
    if byc <= by:
        bxc = bx
    else:
        bxc = by*ff
        byc = by
    ba= int((bx-bxc)/2)
    bb= int(bxc-((bx-bxc)/2))
    bc= int((by-byc)/2)
    bd= int(byc-((by-byc)/2))
    back = bIMG[ba:bb,bc:bd,:]
    return back

    

for i in range (1,350):
    
    backImageName = 'lady_image/JPG_B/' + str(i)+'.jpg'
    ###载入背景图
    foreImageName = 'lady_image/lady_image/' + str(i)+'.jpg'
    ###载入前景图
    maskName = 'lady_image/lady_image_print/' + str(i)+'_seg_color.jpg'
    ###载入二值图
    outputName = 'lady_image/result5/' + str(i)+'.jpg'
    
    backImg = cv2.imread(backImageName,cv2.IMREAD_UNCHANGED) 
    foreImg = cv2.imread(foreImageName,cv2.IMREAD_UNCHANGED) 
    mask = cv2.imread(maskName,0) 
    
    ####背景剪切
    backImg = change_size(foreImg,backImg)
    ###mask处理
    mask = blendingPoisson_bianyuan.change_Bi(mask)

    ###泊松融合时需要加边缘防止移除    
    backImg_posion = backImg / 255.0
    foreImg_posion  = foreImg / 255.0
    mask_posion  = mask / 255.0


    ####更改长宽一致
    backImg = cv2.resize(backImg, (rows,cols))
    foreImg = cv2.resize(foreImg, (rows,cols))
    mask = cv2.resize(mask, (rows,cols))
    backImg_posion = cv2.resize(backImg_posion, (rows,cols))
    foreImg_posion = cv2.resize(foreImg_posion, (rows,cols))
    mask_posion = cv2.resize(mask_posion, (rows,cols))
    
    ####泊松融合处理
    merged_posion = blendingPoisson_alpha.poissonProcess(backImg_posion,foreImg_posion,mask_posion)
    
    ####segment提升精度
    foreImg = blendingPoisson_bianyuan.Original_segmentation(foreImg,mask)
    foreImg ,mask = blendingPoisson_bianyuan.Original_Canny(foreImg,mask)
    foreImg  = cv2.medianBlur(foreImg,5)
    cv2.imwrite(outputName , foreImg)
    
    foreImg = foreImg/255.0
    mask_posion = mask/255.0
    ###segment覆盖
    foreImg_color = [0.6,0.7,0.8,0.9]  ###给定边缘渐变数值
    backImg_color = foreImg_color[::-1]   
    print_merged,WC= blendingPoisson_alpha.slow_color(foreImg_posion,merged_posion,mask_posion,backImg_color,foreImg_color)
    median = print_merged*255
#    median.astype(np.uint8)
#    median = cv2.medianBlur(print_merged,5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(outputName, median)
    
#    
#    videoWriter.write(median)  
#    print (time.time() - start)     
#    videoWriter.release()  
