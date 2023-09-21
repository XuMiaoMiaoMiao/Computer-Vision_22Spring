import cv2
import matplotlib.pyplot as plt

I1 = cv2.imread('pictures/match7.png')
I2 = cv2.imread('pictures/match8.png')

I1 = cv2.cvtColor(I1,cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2,cv2.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.imshow(I1),plt.title('object'),plt.axis ('off') 
plt.subplot(1,2,2)
plt.imshow(I2),plt.title('a scene containing this object'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress