import cv2
import numpy as np
from skimage import transform,color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from skimage.draw import circle_perimeter

I = cv2.imread('pictures/circle_1.jpg',cv2.IMREAD_COLOR)
Iedge = cv2.Canny(I,80,160,None,3)
Igray = cv2.cvtColor(I,cv2.IMREAD_COLOR)

hough_radiu = np.arange(340, 350, 2)
hough_res = transform.hough_circle(Iedge,hough_radiu)
r340 = hough_res[0]
r344 = hough_res[2]
r346 = hough_res[3]
r348 = hough_res[4]
accums, cx, cy, radiu = transform.hough_circle_peaks(hough_res, hough_radiu,total_num_peaks=13)
# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
Iout = I.copy()
for center_y, center_x, radius in zip(cy, cx, radiu):
    cv2.circle(Iout,(center_x,center_y),radius,(0,255,0),10)
    print((center_x,center_y),radius,'\n')
Iout = cv2.cvtColor(Iout,cv2.COLOR_BGR2RGB)
ax.imshow(Iout, cmap=plt.cm.gray)
plt.show()


I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
Iedge = cv2.cvtColor(Iedge,cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.title('The original image.'),plt.axis ('off') 
plt.imshow(I)
plt.subplot(122),plt.title('Source image processed by the Canny algorithm.'),plt.axis ('off') 
plt.imshow(Iedge)
plt.show()
plt.waitforbuttonpress

figure = plt.figure()
axes = figure.add_axes(Axes3D(figure)) 

X = np.arange(0, 2645, 1)
Y = np.arange(0, 2645, 1)
 
X, Y = np.meshgrid(X, Y)
Z = r340
axes.plot_surface(X, Y, Z,cmap='rainbow')
axes.set_title('R = 340')
plt.show()

figure = plt.figure()
axes = figure.add_axes(Axes3D(figure)) 
X = np.arange(0, 2645, 1)
Y = np.arange(0, 2645, 1)
 
X, Y = np.meshgrid(X, Y)
Z = r344
axes.plot_surface(X, Y, Z,cmap='rainbow')
axes.set_title('R = 344')
plt.show()

figure = plt.figure()
axes = figure.add_axes(Axes3D(figure)) 
X = np.arange(0, 2645, 1)
Y = np.arange(0, 2645, 1)
 
X, Y = np.meshgrid(X, Y)
Z = r346
axes.plot_surface(X, Y, Z,cmap='rainbow')
axes.set_title('R = 346')
plt.show()

figure = plt.figure()
axes = figure.add_axes(Axes3D(figure)) 
X = np.arange(0, 2645, 1)
Y = np.arange(0, 2645, 1)
 
X, Y = np.meshgrid(X, Y)
Z = r348
axes.plot_surface(X, Y, Z,cmap='rainbow')
axes.set_title('R = 348')
plt.show()