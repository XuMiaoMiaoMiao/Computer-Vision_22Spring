import cv2
import matplotlib.pyplot as plt
from numpy import zeros_like
import numpy as np
I = cv2.imread('pictures/circle_3.jfif',cv2.IMREAD_COLOR)
#preprocess an image with Canny algorithm:
Iedge= cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
Ib = zeros_like(I)
Iout = I.copy()

circles = cv2.HoughCircles(Iedge,cv2.HOUGH_GRADIENT,1,160,param1=100,param2=79,minRadius=0,maxRadius=0)
print('circles.shape:',circles.shape)
print('circles:\n',circles)
circles = np.uint16(np.around(circles))
for cir in circles[0]:
    print('circle.shape:',cir.shape,'circle:',cir)
    cv2.circle(Ib,(cir[0],cir[1]),cir[2],(0,255,0),10)
    cv2.circle(Ib,(cir[0],cir[1]),2,(0,255,0),3)
    cv2.circle(Iout,(cir[0],cir[1]),cir[2],(0,255,0),10)
    cv2.circle(Iout,(cir[0],cir[1]),2,(0,255,0),3)

I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
Iedge = cv2.cvtColor(Iedge,cv2.COLOR_BGR2RGB)
Ib = cv2.cvtColor(Ib,cv2.COLOR_BGR2RGB)
Iout = cv2.cvtColor(Iout,cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.title('The original image.'),plt.axis ('off') 
plt.imshow(I)
plt.subplot(122),plt.title('Source image processed by the Canny algorithm.'),plt.axis ('off') 
plt.imshow(Iedge)
plt.show()
plt.waitforbuttonpress
plt.subplot(121),plt.title('Hough transform:found circles.'),plt.axis ('off') 
plt.imshow(Ib)
plt.subplot(122),plt.title('Hough transform:found circles on top of the source image.'),plt.axis ('off') 
plt.imshow(Iout)
plt.show()
plt.waitforbuttonpress
'''

Ioutp = cv2.cvtColor(Ioutp,cv2.COLOR_BGR2RGB)
Ipb = cv2.cvtColor(Ipb,cv2.COLOR_BGR2RGB)
plt.imshow(Ih),plt.title(' parameter space.'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress

plt.subplot(121),plt.title('probabilistic Hough transform:found lines.'),plt.axis ('off') 
plt.imshow(Ipb)
plt.subplot(122),plt.title('probabilistic Hough transform:found lines on top of the source image.'),plt.axis ('off') 
plt.imshow(Ioutp)
plt.show()
plt.waitforbuttonpress
'''

