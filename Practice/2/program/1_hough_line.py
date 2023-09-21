import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
I = cv2.imread('pictures/line_3.jpg',cv2.IMREAD_COLOR)
#preprocess an image with Canny algorithm:
Iedge= cv2.Canny(I,200,250,None,3)
#Hough parameter
angles = np.linspace(-np.pi/2,np.pi/2,360,endpoint=False)
Ih,theta,rho = transform.hough_line(Iedge,theta = angles)
Ih = cv2.resize(Ih.astype(np.float32)/np.max(Ih),(Ih.shape[1],400))
#Hough transform
lines = cv2.HoughLines(Iedge,1,np.pi/180,180)
'''
returned value: all (rho,theta)
x*cos(theta) + y*sin(theta) = rho
'''
# probabilistic Hough transform
linesP = cv2.HoughLinesP(Iedge,1,np.pi/180,50,None,50,4)
'''
returned value: the endpoint of the line found
x*cos(theta) + y*sin(theta) = rho
'''
Iout = I.copy()
Ioutp = I.copy()
Ib = np.zeros_like(I)
Ipb = np.zeros_like(I)
#Hough transform(draw lines)
if lines is not None:
    for i in range(0,len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a,b = math.cos(theta) ,math.sin(theta)
        x0 ,y0 = a*rho ,b*rho
        pt1 = np.int32((x0 - 1000*b, y0 + 1000*a))
        pt2 = np.int32((x0 + 1000*b, y0 - 1000*a))
        cv2.line(Iout,pt1,pt2,(0,255,0),2,cv2.LINE_AA)
        cv2.line(Ib,pt1,pt2,(0,255,0),2,cv2.LINE_AA)
#probabilistic Hough transform(draw lines)
min = 10000
max = 0
minn = 0
maxn = 0
number = linesP.shape[0]

if linesP is not None:
    for i in range(0,len(linesP)):
        l = linesP[i][0]
        distance = math.hypot(l[0]-l[2],l[1]-l[3])
        if distance >max:
            max = distance
        if distance < min:
            min = distance
        print((l[0],l[1]),(l[2],l[3]),distance)
        cv2.line(Ioutp,(l[0],l[1]),(l[2],l[3]),(0,255,0),5,cv2.LINE_AA)
        cv2.circle(Ioutp,(l[0],l[1]),2,(0,0,255),5)
        cv2.circle(Ioutp,(l[2],l[3]),2,(0,0,255),5)
        cv2.line(Ipb,(l[0],l[1]),(l[2],l[3]),(0,255,0),5,cv2.LINE_AA)
        cv2.circle(Ipb,(l[0],l[1]),2,(0,0,255),5)
        cv2.circle(Ipb,(l[2],l[3]),2,(0,0,255),5)
I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
Iedge = cv2.cvtColor(Iedge,cv2.COLOR_BGR2RGB)
Ib = cv2.cvtColor(Ib,cv2.COLOR_BGR2RGB)
Iout = cv2.cvtColor(Iout,cv2.COLOR_BGR2RGB)
Ioutp = cv2.cvtColor(Ioutp,cv2.COLOR_BGR2RGB)
Ipb = cv2.cvtColor(Ipb,cv2.COLOR_BGR2RGB)
print('number of lines:',number,'\n')
print('lengths of the longest lines:',min,'\n')
print('lengths of the shortest lines:',max,'\n')

plt.subplot(121),plt.title('The original image.'),plt.axis ('off') 
plt.imshow(I)
plt.subplot(122),plt.title('Source image processed by the Canny algorithm.'),plt.axis ('off') 
plt.imshow(Iedge)
plt.show()
plt.waitforbuttonpress
plt.imshow(Ih),plt.title(' parameter space.'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress
plt.subplot(121),plt.title('Hough transform:found lines.'),plt.axis ('off') 
plt.imshow(Ib)
plt.subplot(122),plt.title('Hough transform:found lines on top of the source image.'),plt.axis ('off') 
plt.imshow(Iout)
plt.show()
plt.waitforbuttonpress
plt.subplot(121),plt.title('probabilistic Hough transform:found lines.'),plt.axis ('off') 
plt.imshow(Ipb)
plt.subplot(122),plt.title('probabilistic Hough transform:found lines on top of the source image.'),plt.axis ('off') 
plt.imshow(Ioutp)
plt.show()
plt.waitforbuttonpress


