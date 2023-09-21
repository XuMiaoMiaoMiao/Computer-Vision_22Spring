import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

# Read an image from file and preprocess
I = cv2.imread('pictures/line_3.jpg',cv2.IMREAD_COLOR)
Iedge = cv2.Canny(I,50,200,None,3)
#Do Hough transform
angles = np.linspace(-np.pi/2,np.pi/2,360,endpoint=False)# -90~90
ps,theta,rho = transform.hough_line(Iedge,theta=angles)
mmin = np.min(rho)
#Find lines with Hough transform
Ih,theta,rho = transform.hough_line_peaks(ps,theta,rho,min_distance=0, min_angle=0, threshold=0.3*np.max(ps))
#Histogram equalization of parameter space
ps = np.array(ps,dtype='uint8')
ps = cv2.equalizeHist(ps)
#Conversion of parameter space to three-channel display 
# (easy to highlight selected points)
img2 = np.zeros((ps.shape[0],360,3),int)
img2[:,:,0] = ps
img2[:,:,1] = ps
img2[:,:,2] = ps
#Create and output image
Iout = I.copy()
Ib = np.zeros_like(I)
if theta is not None:
    for i in range(0,len(theta)):
        a = math.cos(theta[i])
        b = math.sin(theta[i])
        #Highlight selected points in the parameter space
        cv2.circle(img2,((int(round(theta[i]*180/math.pi)+90))*2,int(round(rho[i]-mmin))),2,(0,255,0),10)
        x0,y0 = a*rho[i],b*rho[i]
        pt1 = np.int32((x0-1000*b,y0+1000*a))
        pt2 = np.int32((x0+1000*b,y0-1000*a))
        cv2.line(Iout,pt1,pt2,(0,255,0),2,cv2.LINE_AA)
        cv2.line(Ib,pt1,pt2,(0,255,0),2,cv2.LINE_AA)
#Zoom parameter space image
img3 = np.zeros((400,360,3))
img3[:,:,0] = cv2.resize(img2[:,:,0].astype(np.float32)/np.max(img2[:,:,0]),(img2[:,:,0].shape[1],400))
img3[:,:,1] = cv2.resize(img2[:,:,1].astype(np.float32)/np.max(img2[:,:,1]),(img2[:,:,1].shape[1],400))
img3[:,:,2] = cv2.resize(img2[:,:,2].astype(np.float32)/np.max(img2[:,:,2]),(img2[:,:,2].shape[1],400))
img2 = np.zeros((400,360,3))
ps = cv2.resize(ps.astype(np.float32)/np.max(ps),(ps.shape[1],400))
img2[:,:,0] = ps
img2[:,:,1] = ps
img2[:,:,2] = ps
I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
Iedge = cv2.cvtColor(Iedge,cv2.COLOR_BGR2RGB)
Ib = cv2.cvtColor(Ib,cv2.COLOR_BGR2RGB)
Iout = cv2.cvtColor(Iout,cv2.COLOR_BGR2RGB)
#display image
plt.subplot(121),plt.title('The original image.'),plt.axis ('off') 
plt.imshow(I)
plt.subplot(122),plt.title('Source image processed by the Canny algorithm.'),plt.axis ('off') 
plt.imshow(Iedge)
plt.show()
plt.waitforbuttonpress

plt.subplot(121),plt.title('original parameter space.'),plt.axis ('off') 
plt.imshow(img2)
plt.subplot(122),plt.title('parameter space:Highlight selected point'),plt.axis ('off') 
plt.imshow(img3)
plt.show()
plt.waitforbuttonpress

plt.subplot(121),plt.title('Hough transform:found lines.'),plt.axis ('off') 
plt.imshow(Ib)
plt.subplot(122),plt.title('Hough transform:found lines on top of the source image.'),plt.axis ('off') 
plt.imshow(Iout)
plt.show()
plt.waitforbuttonpress
