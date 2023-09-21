import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    orb = cv2.ORB_create(1000)
    Ifp,Ides = orb.detectAndCompute(Igray,None)
    image_ORB = cv2.drawKeypoints(image,Ifp,None,color = (0,255,0),flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return Ifp, Ides ,image_ORB
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

def point_transfor(p1,Mr2l):
    p1x = (Mr2l[0][0]*p1[0] + Mr2l[0][1]*p1[1] + Mr2l[0][2]) / ((Mr2l[2][0]*p1[0] + Mr2l[2][1]*p1[1] + Mr2l[2][2]))
    p1y = (Mr2l[1][0]*p1[0] + Mr2l[1][1]*p1[1] + Mr2l[1][2]) / ((Mr2l[2][0]*p1[0] + Mr2l[2][1]*p1[1] + Mr2l[2][2]))
    p1_after = (int(p1x), int(p1y)) # after transformation
    return p1_after


def stitch(img_left,left_fp,left_des,img_right,right_fp,right_des):
    #Algorithm Parameter Setting
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
    #matcher = cv2.FlannBasedMatcher(index_params, dict())
    #find KNN matvhes with k = 2
    matcher = cv2.BFMatcher ( crossCheck = False )
    matches = matcher.knnMatch(left_des, right_des, k = 2)
    # Select good matches
    knn_ratio = 0.75
    good = []
    for m in matches:
        if len(m) > 1:
            if m[0].distance < knn_ratio *m[1].distance:
                good.append(m[0])
    matches = good
    num_matches = 100
    matches = sorted(matches,key = lambda x:x.distance)
    Imatch = cv2.drawMatches(img_left,left_fp,img_right,right_fp,matches[:num_matches],None,flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor= (0,255,0))
    #Executing RANSAC to calculate the transformation matrix
    MIN_MATCH_COUNT = 10
    if len(matches) <MIN_MATCH_COUNT:
        print('no enough matches.')
    else:
        #create arrays of point coordinates
        I1pts = np.float32([left_fp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        I2pts = np.float32([right_fp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        #Run RANSAC method
        Mr2l,mask = cv2.findHomography(I2pts,I1pts,cv2.RANSAC,5)
    h_left,w_left = img_left.shape[:2]
    h_right,w_right = img_right.shape[:2]
    #image = np.zeros((max(h_left,h_right),w_left+w_right,3),dtype= 'uint8')
    #image[0:h_right,0:w_right] = img_right
    p1 = (w_right,0) # Coordinates of the upper right point
    p2 =(w_right,h_right) # Coordinates of the lower right point
    p1_after = point_transfor(p1,Mr2l)
    p2_after = point_transfor(p2,Mr2l)
    print(p1_after)
    image = cv2.warpPerspective(img_right,Mr2l,(min(p1_after[0],p2_after[0]),img_left.shape[0]))
    image[0:h_left,0:w_left] = img_left
    return Imatch,image

I1 = cv2.imread('pictures/SIFT9.png')
I2 = cv2.imread('pictures/SIFT10.png')
I3 = cv2.imread('pictures/SIFT11.png')
I1fp_ORB ,I1des_ORB ,I1_ORB = ORB_detect(I1)
I2fp_ORB ,I2des_ORB ,I2_ORB = ORB_detect(I2)
I3fp_ORB,I3des_ORB,I3_ORB = ORB_detect(I3)
I1fp_SIFT,I1des_SIFT,I1_SIFT = SIFT_detect(I1)
I2fp_SIFT,I2des_SIFT,I2_SIFT = SIFT_detect(I2)
I3fp_SIFT,I3des_SIFT,I3_SIFT = SIFT_detect(I3)

Imatch_ORB ,I_stitch_ORB  = stitch(I1,I1fp_ORB ,I1des_ORB ,I2,I2fp_ORB ,I2des_ORB )
Istitch1fp_ORB ,Istitch1des_ORB ,Istitch1_ORB = ORB_detect(I_stitch_ORB )
Isum_ORB ,Isum_stitch_ORB  = stitch(I_stitch_ORB ,Istitch1fp_ORB ,Istitch1des_ORB ,I3,I3fp_ORB,I3des_ORB)
Imatch_SIFT,I_stitch_SIFT = stitch(I1,I1fp_SIFT,I1des_SIFT,I2,I2fp_SIFT,I2des_SIFT)
Istitch1fp_SIFT,Istitch1des_SIFT,Istitch1_SIFT = SIFT_detect(I_stitch_SIFT)
Isum_SIFT,Isum_stitch_SIFT = stitch(I_stitch_SIFT,Istitch1fp_SIFT,Istitch1des_SIFT,I3,I3fp_SIFT,I3des_SIFT)

Isum_stitch_ORB  = cv2.cvtColor(Isum_stitch_ORB ,cv2.COLOR_BGR2RGB)
Isum_ORB  = cv2.cvtColor(Isum_ORB ,cv2.COLOR_BGR2RGB)
Imatch_ORB  = cv2.cvtColor(Imatch_ORB ,cv2.COLOR_BGR2RGB)
I_stitch_ORB  = cv2.cvtColor(I_stitch_ORB ,cv2.COLOR_BGR2RGB)
Isum_stitch_SIFT = cv2.cvtColor(Isum_stitch_SIFT,cv2.COLOR_BGR2RGB)
Isum_SIFT = cv2.cvtColor(Isum_SIFT,cv2.COLOR_BGR2RGB)
Imatch_SIFT = cv2.cvtColor(Imatch_SIFT,cv2.COLOR_BGR2RGB)
I_stitch_SIFT = cv2.cvtColor(I_stitch_SIFT,cv2.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.imshow(Imatch_SIFT),plt.title('match image(SIFT)'),plt.axis ('off') 
plt.subplot(1,2,2)
plt.imshow(Imatch_ORB ),plt.title('match image(ORB)'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress
plt.subplot(1,2,1)
plt.imshow(I_stitch_SIFT),plt.title('stitch image(SIFT)'),plt.axis ('off') 
plt.subplot(1,2,2)
plt.imshow(I_stitch_ORB ),plt.title('match image(ORB)'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress
plt.subplot(2,1,1)
plt.imshow(Isum_SIFT),plt.title('match image(SIFT)'),plt.axis ('off') 
plt.subplot(2,1,2)
plt.imshow(Isum_ORB ),plt.title('match image(ORB)'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress
plt.subplot(1,2,1)
plt.imshow(Isum_stitch_SIFT),plt.title('stitch image(SIFT)'),plt.axis ('off') 
plt.subplot(1,2,2)
plt.imshow(Isum_stitch_ORB ),plt.title('stitch image(ORB)'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress