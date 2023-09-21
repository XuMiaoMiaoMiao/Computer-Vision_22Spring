import cv2
import matplotlib.pyplot as plt
from numpy import zeros_like
import numpy as np
def CIE_lab_kmeans_segmentation(image,n_blocks):
    '''
    n_blocks : The category into which the image needs to be divided
    image : BGR image
    return : segmentedFrames
    segmentedFrames : (RGB)The set of images after segmentation
    '''
    Ilab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    Ilab = cv2.split(Ilab)
    ab = cv2.merge([Ilab[1],Ilab[2]])
    ab = ab.reshape(-1,2).astype(np.float32)###
    #criteria is defined as not more than 10 
    #iterations of difference between steps less than 1. 
    criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    ret,labels,centers = cv2.kmeans(ab,n_blocks,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((Ilab[0].shape))
    segmentedFrames = []
    for i in range(n_blocks):
        Itmp = zeros_like(image)
        mask = labels == i
        Itmp[mask] = image[mask,:]
        Itmp = cv2.cvtColor(Itmp,cv2.COLOR_BGR2RGB)
        segmentedFrames.append(Itmp)
    return segmentedFrames


#input the image
#2 categories
I1 = cv2.imread("pictures/ball2.jpg",cv2.IMREAD_COLOR)
segmentedFrames1 = CIE_lab_kmeans_segmentation (I1,2)
#3 categories
I2 = cv2.imread("pictures/ball3.jpg",cv2.IMREAD_COLOR)
segmentedFrames2 = CIE_lab_kmeans_segmentation (I2,3)
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

plt.subplot(221),plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.imshow(I1)
plt.subplot(222),plt.title('segmented image 1'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[0])
plt.subplot(223),plt.title('segmented image 2'),plt.xticks([]),plt.yticks([])
plt.imshow(segmentedFrames1[1])
plt.show()
plt.waitforbuttonpress

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