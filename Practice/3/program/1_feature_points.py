import cv2
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
    print('The number of feature points detected by SIFT:',len(Ifp))
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
    orb = cv2.ORB_create()
    Ifp,Ides = orb.detectAndCompute(Igray,None)
    image_ORB = cv2.drawKeypoints(image,Ifp,None,color = (0,255,0),flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print('The number of feature points detected by ORB',len(Ifp))
    return Ifp, Ides ,image_ORB
#
I = cv2.imread('41.jpg')
Ifp_SIFT, Ides_SIFT ,I_SIFT = SIFT_detect(I)
Ifp_ORB, Ides_ORB ,I_ORB = ORB_detect(I)
I_SIFT = cv2.cvtColor(I_SIFT,cv2.COLOR_BGR2RGB)
I_ORB = cv2.cvtColor(I_ORB,cv2.COLOR_BGR2RGB)
plt.subplot(121),plt.title('SIFT detector'),plt.axis ('off') 
plt.imshow(I_SIFT)
plt.subplot(122),plt.title('ORB detector'),plt.axis ('off') 
plt.imshow(I_ORB)
plt.show()
plt.waitforbuttonpress
