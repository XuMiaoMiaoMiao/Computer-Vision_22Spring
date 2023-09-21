import cv2
import numpy as np
import matplotlib.pyplot as plt
def SIFT_detect(image):
    '''
    Ifp,Ides,iamge_SIFT = SIFT_detect(image)
    image : BGR image  
    Ifp : feature points 
    Ides : feature points corresponding descriptors
    imageSIFT :image displaying SIFT feature points 
    in green color with scale and orientation
    '''
    Igray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    Ifp, Ides = sift.detectAndCompute(Igray,None)
    image_SIFT =image.copy()
    image_SIFT = cv2.drawKeypoints(image,Ifp,image_SIFT,color = (0,255,0),flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return Ifp, Ides ,image_SIFT


def ORB_detect(image):
    '''
    Ifp,Ides,iamge_SIFT = SIFT_detect(image)
    image : BGR image  
    Ifp : feature points 
    Ides : feature points corresponding descriptors
    imageSIFT :image displaying SIFT feature points 
    in green color with scale and orientation
    '''
    Igray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(10000)
    Ifp,Ides = orb.detectAndCompute(Igray,None)
    image_ORB = cv2.drawKeypoints(image,Ifp,None,color = (0,255,0),flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return Ifp, Ides ,image_ORB

def knn_RANSAC_match(I1,I1fp,I1des,I2,I2fp,I2des):
    '''
    Imatch,Itrans = SIFT_ANNmatch(I1,I1fp,I1des,I2,I2fp,I2des)
    I1,I2 :BGR original image
    I1fp,I2fp : feature points 
    I1des, I2des : feature points corresponding descriptors
    Imatch : 100 strongest matches(FLANN)
    Itrans : Inliers found by the RANSAC method
    '''
    #Algorithm Parameter Setting
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
    #matcher = cv2.FlannBasedMatcher(index_params, dict())
    matcher = cv2.BFMatcher ( crossCheck = False )
    #find KNN matches with k = 2
    matches = matcher.knnMatch(I1des, I2des, k = 2)
    # Select good matches
    knn_ratio = 0.8
    good = []
    for m in matches:
        if len(m) > 1:
            if m[0].distance < knn_ratio *m[1].distance:
                good.append(m[0])
    matches = good
    # Displaying top 50 matches
    num_matches = 50
    matches = sorted(matches,key = lambda x:x.distance)
    Imatch = cv2.drawMatches(I1,I1fp,I2,I2fp,matches[:num_matches],None,flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor= (0,255,0))
    #Executing RANSAC to calculate the transformation matrix
    MIN_MATCH_COUNT = 10
    if len(matches) <MIN_MATCH_COUNT:
        print('no enough matches.')
    else:
        #create arrays of point coordinates
        I1pts = np.float32([I1fp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        I2pts = np.float32([I2fp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        #Run RANSAC method
        M,mask = cv2.findHomography(I1pts,I2pts,cv2.RANSAC,5)
        mask = mask.ravel().tolist()
    h,w = I1.shape[:2]
    I1box = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    I1to2box = cv2.perspectiveTransform(I1box,M)
    #Draw a red box on the second image
    I2res = cv2.polylines(I2,[np.int32(I1to2box)],True,(0,0,255),3,cv2.LINE_AA)
    Itrans = cv2.drawMatches(I1,I1fp,I2res,I2fp,matches,None,matchesMask= mask,flags= cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor=(0,255,0))
    return Imatch,Itrans,I2res

#ORB

I11 = cv2.imread('pictures/match7.png')
I21 = cv2.imread('pictures/match8.png')
I12 = cv2.imread('pictures/match7.png')
I22 = cv2.imread('pictures/match8.png')

I1fp_SIFT,I1des_SIFT,I1_SIFT = SIFT_detect(I11)
I2fp_SIFT,I2des_SIFT,I2_SIFT = SIFT_detect(I21)
I1fp_ORB,I1des_ORB,I1_ORB = ORB_detect(I12)
I2fp_ORB,I2des_ORB,I2_ORB = ORB_detect(I22)

Imatch_SIFT,Itrans_SIFT,Ires_SIFT = knn_RANSAC_match(I11,I1fp_SIFT,I1des_SIFT,I21,I2fp_SIFT,I2des_SIFT)
Imatch_ORB,Itrans_ORB,Ires_ORB = knn_RANSAC_match(I12,I1fp_ORB,I1des_ORB,I22,I2fp_ORB,I2des_ORB)

Imatch_SIFT = cv2.cvtColor(Imatch_SIFT,cv2.COLOR_BGR2RGB)
Itrans_SIFT = cv2.cvtColor(Itrans_SIFT,cv2.COLOR_BGR2RGB)
Ires_SIFT = cv2.cvtColor(Ires_SIFT,cv2.COLOR_BGR2RGB)
Imatch_ORB = cv2.cvtColor(Imatch_ORB,cv2.COLOR_BGR2RGB)
Itrans_ORB = cv2.cvtColor(Itrans_ORB,cv2.COLOR_BGR2RGB)
Ires_ORB = cv2.cvtColor(Ires_ORB,cv2.COLOR_BGR2RGB)

plt.subplot(2,1,1)
plt.imshow(Imatch_SIFT),plt.title('50 strongest matches with FLANN method and SIFT descriptors'),plt.axis ('off') 
plt.subplot(2,1,2)
plt.imshow(Imatch_ORB),plt.title('50 strongest matches with brute force method and ORB descriptors'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress
plt.subplot(2,1,1)
plt.imshow(Itrans_SIFT),plt.title('Inliers found by the RANSAC method(SIFT,kd-trees,knn,RANSAC)'),plt.axis ('off') 
plt.subplot(2,1,2)
plt.imshow(Itrans_ORB),plt.title('Inliers found by the RANSAC method(ORB,BF,knn,RANSAC)'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress
plt.subplot(1,2,1)
plt.imshow(Ires_SIFT),plt.title('Detected object(SIFT,kd-trees,knn,RANSAC)'),plt.axis ('off') 
plt.subplot(1,2,2)
plt.imshow(Ires_ORB),plt.title('Detected object(ORB,BF,knn,RANSAC)'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress



