import cv2
from cv2 import sqrt
import matplotlib.pyplot as plt
from numpy import zeros_like
import numpy as np
def CIE_lab_nearest_segmentation(image,n_blocks):
    '''
    n_blocks : The category into which the image needs to be divided
    image : BGR image
    return : segmentedFrames,Iplot
    segmentedFrames : (RGB)The set of images after segmentation
    Iplot : (RGB)The distribution of segmented image on axis (a,b)
    '''
    Ilab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    Ilab = cv2.split(Ilab)
    #Color sample space generation
    sampleAreas = []
    def MouseHandler(event,x,y,flags,param):
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return
        sampleAreas.append((x,y))
    cv2.imshow("image",image)
    cv2.setMouseCallback('image',MouseHandler)
    while len(sampleAreas) < n_blocks:
        cv2.waitKey(20)
    cv2.setMouseCallback('image',lambda*args:None)
    #Calculate average color
    colorMarksLAB = []
    colorMarksBGR = []
    for pix in sampleAreas:
        mask = zeros_like(Ilab[0])
        cv2.circle(mask,pix,10,255,-1)
        a = np.mean(Ilab[1][np.argwhere(mask > 0)])
        b = np.mean(Ilab[2][np.argwhere(mask > 0)])
        colorMarksLAB.append((a,b))
        colorMarksBGR.append(image[mask>0,:].mean(axis=(0)))
    #Calculates the minimum distance between all pixels and all colors
    distance = []
    for color in colorMarksLAB:
        distance.append(sqrt(pow((Ilab[1]-color[0]),2)+pow((Ilab[2]-color[1]),2)))
    distance_min = np.minimum.reduce(distance)
    #Calculates the color label for each pixel
    labels = zeros_like(Ilab[0],dtype= np.uint8)
    for i in range(len(colorMarksLAB)):
        mask = distance_min ==distance[i]
        labels[mask] = i
    #Split and store the original image
    segmentedFrames = []
    for i in range(len(colorMarksLAB)):
        Itmp = zeros_like(image)
        mask = labels == i
        Itmp[mask] = image[mask]
        segmentedFrames.append(Itmp)
    #Displays the color distribution in (a,b) coordinates
    Iplot = np.full((256,256,3),255,dtype=np.uint8)
    for i in range(len(colorMarksLAB)):
        Itmp = zeros_like(image)
        mask = labels == i
        Iplot[Ilab[1][mask],Ilab[2][mask],:] = colorMarksBGR[i]
    for i in range(n_blocks):
        segmentedFrames[i] = cv2.cvtColor(segmentedFrames[i],cv2.COLOR_BGR2RGB)
    Iplot = cv2.cvtColor(Iplot,cv2.COLOR_BGR2RGB)
    return segmentedFrames,Iplot

#input the image
#2 categories
I1 = cv2.imread("pictures/ball2.jpg",cv2.IMREAD_COLOR)
segmentedFrames1,Iplot1 = CIE_lab_nearest_segmentation(I1,2)
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)

plt.subplot(221),plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.imshow(I1)
plt.subplot(222),plt.title('segmented image 1'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[0])
plt.subplot(223),plt.title('segmented image 2'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[1])
plt.show()
plt.waitforbuttonpress

#3 categories
I2 = cv2.imread("pictures/ball3.jpg",cv2.IMREAD_COLOR)
segmentedFrames2,Iplot2 = CIE_lab_nearest_segmentation(I2,3)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

plt.subplot(221),plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.imshow(I2)
plt.subplot(222),plt.title('segmented image 1'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames2[0])
plt.subplot(223),plt.title('segmented image 2'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames2[1])
plt.subplot(224),plt.title('segmented image 3'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames2[2])
plt.show()
plt.waitforbuttonpress