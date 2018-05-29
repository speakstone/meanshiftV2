# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:10:40 2018

@author: lilei0129
"""
from __future__ import print_function

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

  
#def addTransparency(img, factor = 0.7 ):   #增加alpha通道
#    img = img.convert('RGBA')  
#    img_blender = Image.new('RGBA', img.size, (0,0,0,0))  
#    img = Image.blend(img_blender, img, factor)  
#    return img  


def generateData(back,fore,mask):
    I = []
    J = []
    S = []
    B = []
    count = 0

    for i in xrange(int(back.shape[0])):
        for j in xrange(int(back.shape[1])):
            if mask.item(i,j) < 0.1: #black pixel, then insert 1 at that index
                I.extend([count])
                J.extend([count])
                S.extend([1])
                B.extend([back[i,j]]) #set b = background pixel value
            else: #white pixel, insert gradient of i,j
                J.extend([count-1, count+1, count-fore.shape[1], count+fore.shape[1], count])
                I.extend([count,   count,   count,               count,               count])
                S.extend([1,      1,      1,                  1,                   -4])

#		B.extend( [fore[i-1,j] + fore[i+1,j] + fore[i,j-1] + fore[i,j+1] - 4.0*fore[i,j]] )
      
                ''' 
                With Gradient Mixing 
                tmpforeB = fore[i-1,j] + fore[i+1,j] + fore[i,j-1] + fore[i,j+1] - 4.0*fore[i,j]
                tmpbackB = back[i-1,j] + back[i+1,j] + back[i,j-1] + back[i,j+1] - 4.0*back[i,j]
                B.extend( [0.5*tmpforeB + 0.5*tmpbackB] )
                '''
                tmpforeB = fore[i-1,j] + fore[i+1,j] + fore[i,j-1] + fore[i,j+1] - 4.0*fore[i,j]
                tmpbackB = back[i-1,j] + back[i+1,j] + back[i,j-1] + back[i,j+1] - 4.0*back[i,j]
                B.extend( [1.3*tmpforeB + 0*tmpbackB] )
                
            count+=1

    I = np.asarray(I) #column for SPARSE MATRIX A
    J = np.asarray(J) #row for SPARSE MATRIX A
    S = np.asarray(S) #data for SPARSE MATRIX A
    B = np.asarray(B)#B for Ax=b   
    

    return I,J,S,B
    

''' genereateAB using zero ndarray : TOO MUCH MEMORY
def generateAB(back,fore,mask,alls):
    A = np.zeros(shape =(alls,alls))
    B = []
    for i in xrange(int(back.shape[0])):
        for j in xrange(int(back.shape[1])):
            if mask.item(i,j) < 0.5:
                A.itemset((i,j),1)
                B.append(back[i,j])
            else:
                A.itemset((i,j),-4)
                A.itemset((i,j-4),1)
                A.itemset((i,j-1),1)
                A.itemset((i,j+1),1)
                A.itemset((i,j+4),1)
                B.append( [fore[i-1,j] + fore[i,j-1] + fore[i,j+1] + fore[i+1,j] - 4*(fore[i,j])] )
    B = np.asarray(B)
    return A, B
'''
def slow_color(back,fore,mask,backImg_color,foreImg_color,wc=0.6):
    ##输入权重系数
    backImg_color = np.array(backImg_color) 
    foreImg_color = np.array(foreImg_color)
    length = len(foreImg_color)
    
    ##更改只读模式
    back.flags.writeable = True  
    fore.flags.writeable = True 
    
    ##获取长宽
    image_Row = (fore.shape)[0]
    image_Column = (fore.shape)[1]
    
    ##生成权重矩阵
    WC = np.zeros([image_Row,image_Column]) 
    WC_2 = np.ones([image_Row,image_Column]) 
    
    ###第一次生成处理
    for Row in range(1,image_Row):
        for Column in range(1,image_Column):
            if mask[Row,Column] > 0.5:   
                WC[Row,Column] = wc
                
    ###第二次腐蚀膨胀生成处理 
    erosion = mask
    for i in range(length):  
        
        WC_test = WC
        for Row in range(1,image_Row-1):
            for Column in range(1,image_Column-1):
                if erosion[Row-1,Column]- erosion[Row,Column] <0 or erosion[Row,Column-1]- erosion[Row,Column] < 0 :
                   WC_test[Row,Column] =  WC[Row,Column]*foreImg_color[i]
                elif erosion[Row+1,Column]- erosion[Row,Column] <0 or erosion[Row,Column+1]- erosion[Row,Column] < 0 :
                   WC_test[Row,Column] =  WC[Row,Column]*foreImg_color[i]
        kernel = np.ones((5,5),np.uint8)    
        erosion = cv2.erode(erosion,kernel,iterations = 1) 
        WC = WC_test
    
       
#    for Row in range(length,image_Row-length):
#        for Column in range(length,image_Column-length):        
#            if mask[Row,Column] - mask[Row+1,Column] > 0.1:   
#                WC[Row-length+1:Row+1,Column] = wc * backImg_color
#            elif mask[Row,Column] - mask[Row+1,Column] <= (-0.1): 
#                WC[Row-1:Row+length-1,Column] = wc * foreImg_color
#            else:continue
#        
#    for Row in range(length,image_Row-length):
#        for Column in range(length,image_Column-length):
#            if mask[Row,Column+1] - mask[Row,Column+1] > 0.1:   
#                WC[Row,Column-length+1:Column+1] = wc * backImg_color
#            elif mask[Row,Column] - mask[Row,Column+1] <= (-0.1): 
#                WC[Row,Column-length-1:Column-1] = wc * foreImg_color    
#            else:continue   
        
#    for Row in range(1,image_Row):
#        for Column in range(1,image_Column):
#            if WC2[Row,Column] == 0: WC[Row,Column] = 0
                   
    WC_3 = WC_2[:,:]-WC[:,:]     
    fore[:,:,0] = (fore[:,:,0]*(WC_3[:,:]) + back[:,:,0]*(WC[:,:]))
    fore[:,:,1] = fore[:,:,1]*(WC_3[:,:]) + back[:,:,1]*WC[:,:]
    fore[:,:,2] = fore[:,:,2]*(WC_3[:,:]) + back[:,:,2]*WC[:,:]
    
    return fore,WC
    
    
#    ###替换背景
#    for Row in range(1,image_Row):
#        for Column in range(1,image_Column):
#            if mask[Row,Column] > 0.5:             
##                gengxin_Matrix[Row,Column,0] = beijing_Matrix[Row,Column,0]
##                gengxin_Matrix[Row,Column,1] = beijing_Matrix[Row,Column,1]
##                gengxin_Matrix[Row,Column,2] = beijing_Matrix[Row,Column,2]
#                fore[Row,Column,0] = fore[Row,Column,0]*wc+back[Row,Column,0]*(1-wc)
#                fore[Row,Column,1] = fore[Row,Column,1]*wc+back[Row,Column,1]*(1-wc)
#                fore[Row,Column,2] = fore[Row,Column,2]*wc+back[Row,Column,2]*(1-wc)
#    return fore
                    
    
    

# Topic = 'snow'
#Topic = '201'

for i in range (0,376):
    
    backImageName = '1/' + 'beijing2.jpg'   
    ###载入背景图
    foreImageName = 'lady_image/lady_image/' +str(i)+'.jpg'
    ###载入前景图
    maskName = 'lady_image/lady_image_BI/' + 'BI_'+ str(i)+'_seg_color.jpg'
    ###载入二值图
    outputName = 'lady_image/result3/' + str(i)+'.jpg'
        
#    backImageName = '1/' + 'beijing2.jpg'   
#    ###载入背景图
#    foreImageName = '1/' +'0.jpg'
#    ###载入前景图
#    maskName = '1/' +'BI_0_seg_color.jpg'
#    ###载入二值图
#    outputName = '1/' +'print2.jpg'
#        
#   在原图的基础上增加缓冲带处理
#    backImg = cv2.imread(backImageName) 
#    foreImg = cv2.imread(foreImageName) 
#    mask = cv2.imread(maskName) 
#    ##更改只读模式
#    backImg.flags.writeable = True 
#    foreImg.flags.writeable = True 
#    mask.flags.writeable = True 
    
    ##原图上增加缓冲
#    backImg_color = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    

    
    backImg = cv2.imread(backImageName,cv2.IMREAD_UNCHANGED) / 255.0
    foreImg = cv2.imread(foreImageName,cv2.IMREAD_UNCHANGED) / 255.0
    mask = cv2.imread(maskName,0) / 255.0
    
    rows = backImg.shape[0] #545p
    cols = backImg.shape[1] #429p
    channels = backImg.shape[2] #3 for BGR
    
    #alls = rows * cols * channels
    alls = rows * cols #total number of pixels in image
    
    #split BGR
    backB, backG, backR = cv2.split(backImg)
    foreB, foreG, foreR = cv2.split(foreImg)
    
    #print (backImg)
    
    
    ''' TEST NDARRAY ITEMSET
    B = np.zeros(4)
    B.itemset(1,2)
    print B
    '''
    
    ''' TESTING R WITH SMALL NDARRAY 
    print ("***** Testing R with Small NDARRAY *****")
    TestA = np.array([[0,0,1,1],[1,0,2,0],[1,1,0,1],[1,1,0,0]])
    TestB = np.array([15,12,22,16])
    
    print ("matrix A at [2,1]",end ="")
    print (TestA[2,1])
    
    TestA = scipy.sparse.coo_matrix(TestA) #convert np.array A to coo_matrix
    TestA = TestA.tocsc()
    
    print ("matrix A shape: ",end="")
    print (TestA.shape)
    print (TestA)
    print ("matrix B shape: %s" % TestB.shape)
    print (TestB)
    R = scipy.sparse.linalg.spsolve(TestA,TestB)
    print ("solution X type: %s" % type(R))
    print (R)
    print ("solution X shape: ",end="")
    print (R.shape)
    print ("\n\n")
    '''
    
    """
    Construct matrix A & B
    """
       # print ("***** Generating Matrices Ab, Ag, Ar *****")
    
    numRowsInA = alls # pixels(row) * pixels(col)
    
    Ib,Jb,Sb,Bb = generateData(backB,foreB,mask)
    
    Ig, Jg, Sg, Bg = generateData(backG,foreG,mask)
    Ir, Jr, Sr, Br = generateData(backR,foreR,mask)
    
    Ab = scipy.sparse.coo_matrix((Sb, (Ib, Jb)), shape=(numRowsInA, alls))
    Ag = scipy.sparse.coo_matrix((Sg, (Ig, Jg)), shape=(numRowsInA, alls))
    Ar = scipy.sparse.coo_matrix((Sr, (Ir, Jr)), shape=(numRowsInA, alls))
    Ab = Ab.tocsc() # Convert A matrix to Compressed Sparse Row format
    Ag = Ag.tocsc()
    Ar = Ar.tocsc()
    
    """
    extract final result from R
    Solve Ax = b for each of B,G,R
    """
    
    #print ("***** Solving X for AX = B *****")
    #R = scipy.sparse.linalg.cg(Ab, Bb)
    Rb = scipy.sparse.linalg.spsolve(Ab,Bb)
    Rb = np.reshape(Rb, (rows,cols))
    Rg = scipy.sparse.linalg.spsolve(Ag,Bg)
    Rg = np.reshape(Rg, (rows,cols))
    Rr = scipy.sparse.linalg.spsolve(Ar,Br)
    Rr = np.reshape(Rr, (rows,cols))
    merged = cv2.merge((Rb,Rg,Rr))
    
    """
    利用原图增强真实感
    
    """
    
    ##图片增加alpha通道
#    foreImg = addTransparency(foreImg, factor =1.0) 
#    merged = addTransparency(merged, factor =1.0) 
 
#    foreImg_color =np.array(range(5,10,2))*0.1
    foreImg_color = [0.4,0.61,0.89]
    backImg_color = foreImg_color[::-1]   
    print_merged,WC= slow_color(foreImg,merged,mask,backImg_color,foreImg_color)
    
    
    
    #cv2.imshow("merged",merged)
    
    file=open('1/data2.txt','w')  
    file.write(str(WC ));  
    file.close() 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(outputName, print_merged*255)
    
    print (time.time() - start)
    """
    uncomment these lines after you generate the final result in matrix 'img'
    cv2.imshow('output', R);
    cv2.waitKey(0)
    cv2.imwrite(outputName, R * 255);
    """
