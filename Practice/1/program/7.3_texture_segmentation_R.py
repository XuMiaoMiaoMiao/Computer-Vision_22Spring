from ctypes import sizeof
from turtle import st
import cv2
from cv2 import sqrt
import matplotlib.pyplot as plt
from numpy import zeros_like
import numpy as np
from skimage import morphology,filters
def bwareaopen(A,dim,conn= 8):
    if A.ndim != 2 or A.dtype != np.uint8:
        return None
    # Find all connected components
    num,labels,stats,centers = cv2.connectedComponentsWithStats(A,connectivity=conn)
    #check siz of all aonnected components
    for i in range(num):
        if stats[i,cv2.CC_STAT_AREA] <dim:
            A[labels == i] = 0
    return A
def imfillholes(I):
    if I.ndim != 2 or I.dtype !=np.uint8:
        return None
    rows,cols = I.shape[0:2]
    mask = I.copy()
    #Fill mask from all horizontal borders
    for i in range(cols):
        if mask[0,i] == 0:
            cv2.floodFill(mask,None,(i,0),255,10,10)
        if mask[rows-1,i] == 0:
            cv2.floodFill(mask,None,(i,rows-1),255,10,10)
    #Fill mask from all vertical borders
    for i in range(rows):
        if mask[i,0] == 0:
            cv2.floodFill(mask,None,(0,i),255,10,10)
        if mask[i,cols-1] == 0:
            cv2.floodFill(mask,None,(cols-1,i),255,10,10)
    #use the mask to create a resulting image
    res = I.copy()
    res[mask == 0] = 255
    return res

def textures_segmentation_smoothness(image):
    '''
    image :(RGB)original image
    '''
    segmentedFrames = []
    ###############binaryzation##########
    Igray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #Generate a 9x9 rectangular kernel
    kernel = morphology.square(9)
    #Calculate the local smoothness R of the image
    Igray = Igray / 255.0
    Igray_2 = Igray ** 2
    Igray_2_mean = cv2.blur(Igray_2,(9,9))
    Igray_mean = cv2.blur(Igray,(9,9))
    Igray_mean_2 = Igray_mean ** 2
    local_variance = np.maximum((Igray_2_mean - Igray_mean_2),0)
    local_variance = local_variance *255*255
    m,n = Igray.shape
    smooth_R = zeros_like(Igray)
    print(m,n)
    for i in range(m):
        for j in range(n):
            smooth_R[i,j] = 1 - 1/(1+local_variance[i,j])
    #Normalize the obtained float32 value 
    #and convert it to the range of 0~255
    smooth_Rim = np.uint8(smooth_R*255)
    #Binarize the image using 
    #the Otsu thresholding method
    ret,BW1 = cv2.threshold(smooth_Rim,0,255,cv2.THRESH_OTSU)
    segmentedFrames.append(BW1)
    ###############morphological filtering##########
    #1. remove connected areas containing 
    #less than a given number of pixels
    BW1 = bwareaopen(BW1,10000,conn= 8)
    segmentedFrames.append(BW1)
    #2. remove internal form defects or «holes»
    nhood = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    BW1 = cv2.morphologyEx(BW1,cv2.MORPH_CLOSE,nhood)
    segmentedFrames.append(BW1)
    #3. fill the remaining large «holes»
    BW1 = imfillholes(BW1)
    segmentedFrames.append(BW1)
    contours,h = cv2.findContours(BW1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    boundary = zeros_like(BW1)
    cv2.drawContours(boundary,contours,-1,255,3)
    segmentResults = image.copy()
    segmentResults[boundary != 0,0] = 255
    segmentResults[boundary != 0,1] = 0
    segmentResults[boundary != 0,2] = 0
    segmentedFrames.append(segmentResults)
    return segmentedFrames

I1 = cv2.imread('pictures/sea.jpg',cv2.IMREAD_COLOR)
I1 = cv2.cvtColor(I1,cv2.COLOR_BGR2RGB)
segmentedFrames1 = textures_segmentation_smoothness(I1)


plt.subplot(321),plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.imshow(I1,cmap="gray")
plt.subplot(322),plt.title('After filtering and binarization'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[0],cmap="gray")
plt.subplot(323),plt.title('After bwareaopen ()'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[1],cmap="gray")
plt.subplot(324),plt.title('After imclose() '),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[2],cmap="gray")
plt.subplot(325),plt.title('After imfill()'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[3],cmap="gray")
plt.subplot(326),plt.title('Result image'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[4])
plt.suptitle('the features of smoothness: relative smoothness R')
plt.show()
plt.waitforbuttonpress
