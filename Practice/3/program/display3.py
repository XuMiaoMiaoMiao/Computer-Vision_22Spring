import cv2
import matplotlib.pyplot as plt

I1 = cv2.imread('pictures/SIFT9.png')
I2 = cv2.imread('pictures/SIFT10.png')
I3 = cv2.imread('pictures/SIFT11.png')

I1 = cv2.cvtColor(I1,cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2,cv2.COLOR_BGR2RGB)
I3 = cv2.cvtColor(I3,cv2.COLOR_BGR2RGB)

plt.subplot(1,3,1)
plt.imshow(I1),plt.title('left image'),plt.axis ('off') 
plt.subplot(1,3,2)
plt.imshow(I2),plt.title('middle image'),plt.axis ('off') 
plt.subplot(1,3,3)
plt.imshow(I3),plt.title('right image'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress