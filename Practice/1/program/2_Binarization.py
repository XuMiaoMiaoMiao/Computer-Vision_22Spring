import cv2
import matplotlib.pyplot as plt



I = cv2.imread("pictures/Spongebob.jpg",cv2.IMREAD_COLOR)
#I = cv2.imread("pictures/lightning.PNG",cv2.IMREAD_COLOR)

Igray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
histsize=256
histrange=(0,256)
hist = cv2.calcHist([Igray],[0],None,[256],[0,255])
Inew = cv2.adaptiveThreshold(Igray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret,Inew1 = cv2.threshold (Igray,0,255,cv2.THRESH_OTSU)

I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
Inew = cv2.cvtColor(Inew, cv2.COLOR_BGR2RGB)
Inew1 = cv2.cvtColor(Inew1, cv2.COLOR_BGR2RGB)

#Display original image and histogram
plt.subplot(121),plt.title('original image'),plt.axis ('off') 
plt.imshow(I)
plt.subplot(122),plt.title('Greyscale Image Histogram'),plt.axis ('off') 
plt.plot(hist)
plt.show()
plt.waitforbuttonpress

#Display the image after the two methods are changed
plt.subplot(122),plt.title('Adaptive methods'),plt.axis ('off') 
plt.imshow(Inew)
plt.subplot(121),plt.title('OTSU'),plt.axis ('off') 
plt.imshow(Inew1)
plt.show()
plt.waitforbuttonpress
