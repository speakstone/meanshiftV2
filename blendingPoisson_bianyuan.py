# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:43:07 2018

@author: lilei0129
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:10:40 2018

@author: lilei0129
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
np.set_printoptions(threshold=np.inf) 
# config & input
start = time.time()
np.set_printoptions(threshold='nan')  #全部输出  

def change_Bi(image):  ####过滤segment，增加边缘
    img = image
#    img = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
#    img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR)
     ##更改只读模式
    image.flags.writeable = True  
    
    ##获取长宽
    image_Matrix = image[:,:]
    image_Row = (image_Matrix.shape)[0]
    image_Column = (image_Matrix.shape)[1]

    ##改变颜色为黑白二值图,过滤不需要的前景
    for i in range(image_Row):
        for j in range(image_Column):
#            if 150<image_Matrix[Row,Column] < 210 or 100<image_Matrix[Row,Column] <200 or 100<image_Matrix[Row,Column] <200 :
#                image_Matrix[Row,Column] =image_Matrix[Row,Column]=image_Matrix[Row,Column] =255
#            else:
#                image_Matrix[Row,Column] =image_Matrix[Row,Column]=image_Matrix[Row,Column] =0
            if 130>img[i,j] or img[i,j]>200: img[i,j] = 0
            else: img[i,j] = 255
    seg_map = cv2.medianBlur(img,5)
#    Grayimg = cv2.cvtColor(image_Matrix, 0)
#    ret, thresh = cv2.threshold(Grayimg, 12, 255,cv2.THRESH_BINARY) 
    cv2.rectangle(seg_map,(0,0),(image_Column-1,image_Row-1),(0,0),1)
    return seg_map

def Original_Canny(fore,mask):       
    fore_canny = cv2.GaussianBlur(fore,(3,3),0)  
    fore_canny = cv2.Canny(fore_canny, 50, 150)  
    image_Row = (fore_canny.shape)[0]
    image_Column = (fore_canny.shape)[1] 

    
    sum_prefix = np.zeros(fore_canny.shape)
    sum_suffix = np.zeros(fore_canny.shape)
    for i in range(1, image_Row - 1):
        for j in range(1, image_Column - 1):
            sum_prefix[i, j] = sum_prefix[i, j - 1] + fore_canny[i, j - 1]
            sum_suffix[i, image_Column - 1 - j] = sum_suffix[i, image_Column - j] + fore_canny[i, image_Column - j]
    
    for i in range(1, image_Row-1):
        for a in range(1, image_Column-1):
            for b in range(a + 1, min(a + 10, image_Column - 1)):
                if (fore_canny[i, a] == fore_canny[i, b] == 255):
                    if (sum_prefix[i, a] == 0) or (sum_suffix[i, b] == 0):
                        fast = min(a, b)
                        last = abs(a - b) + fast
                        fore[i, fast:last, :] = np.zeros([abs(a - b), 3])
                        mask[i, fast:last] = np.zeros([abs(a - b)])
                        break
#                if (fore_canny[i,a] == fore_canny[i,b] == 255) and (0 < abs(a-b) < 10) and  (sum(fore_canny[i,b+1:]) == 0) :
#                    fast = min(a,b)
#                    last = abs(a-b)+fast
#                    fore[i,fast:last,:] = np.zeros([abs(a-b),3])
#                    c = c+1
#                    d += 1
#                    break

    return fore , mask


 
def Original_segmentation(fore,mask):
    
    ##更改只读模式 
    fore.flags.writeable = True 
    
    ##获取长宽
    image_Row = (fore.shape)[0]
    image_Column = (fore.shape)[1]
                    
    ###腐蚀膨胀生成处理 
#    erosion = mask
#    for i in range(2):        
#        kernel = np.ones((5,5),np.uint8)    
#        erosion = cv2.dilate(erosion,kernel,iterations = 1) 
#        mask = erosion
    
    ###取出图像
    for Row in range(image_Row):
        for Column in range(image_Column):
            if mask[Row,Column] > 100: 
                mask[Row,Column] = 255
                fore[Row,Column,:] = fore[Row,Column,:]
            else:
                mask[Row,Column] = 0
                fore[Row,Column,:] = np.array([0,0,0])
    return fore
    

#
#for i in range(1,200):
##    backImageName = '1/' + 'beijing2.jpg'   
##    ###载入背景图
#    foreImageName = 'lady_image/lady_image/' +str(i)+'.jpg'
#    ###载入前景图
#    maskName = 'lady_image/lady_image_BI/' + 'BI_'+ str(i)+'_seg_color.jpg'
#    ###载入二值图
#    outputName = 'lady_image/lady_image_division/' + str(i)+'.jpg'
#    
#    outputName2 = 'lady_image/lady_image_BI_canny/'+ 'BI_'+str(i)+'.jpg'
#
#    foreImg = cv2.imread(foreImageName,cv2.IMREAD_UNCHANGED) 
#    mask = cv2.imread(maskName,0) 
#    
#    print_merged = Original_segmentation(foreImg,mask)
##    Canny_merged ,mask_merged = Original_Canny(print_merged,mask)
##    median = cv2.medianBlur(Canny_merged,5)
##    print Canny_merged
##    Canny_merged ,c ,fore_canny= Original_Canny(print_merged)
##    print c 
###    cv2.imshow('Canny', Canny_merged) 
#    cv2.imwrite(outputName, print_merged)
##    cv2.imwrite(outputName2, mask_merged)
##    cv2.imwrite('lady_image/test2.jpg', fore_canny)     
##    cv2.waitKey(0)
#    cv2.destroyAllWindows()
##    cv2.imwrite(outputName, print_merged)
##    cv2.imshow(outputName, print_merged*255)    
#    print (time.time() - start)
