import cv2
import matplotlib.pyplot as plt



#I = cv2.imread("pictures/sunflower.jpg",cv2.IMREAD_COLOR)
I = cv2.imread("pictures/Starry_Night.jpg",cv2.IMREAD_COLOR)

Igray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
histsize=256
histrange=(0,256)
hist = cv2.calcHist([Igray],[0],None,[256],[0,255])
ret,Inew = cv2.threshold(Igray,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
ret,Inew1 = cv2.threshold (Igray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
Inew = cv2.cvtColor(Inew, cv2.COLOR_BGR2RGB)
Inew1 = cv2.cvtColor(Inew1, cv2.COLOR_BGR2RGB)

#Display original image and histogram
plt.subplot(121),plt.title('original image'),plt.axis ('off') 
plt.imshow(I)
plt.subplot(122),plt.title('Greyscale Image Histogram')
plt.plot(hist)
plt.show()
plt.waitforbuttonpress

#Display the image after the two methods are changed
plt.subplot(122),plt.title(' use lower threshold binarization.'),plt.axis ('off') 
plt.imshow(Inew)
plt.subplot(121),plt.title(' use upper threshold binarization.'),plt.axis ('off') 
plt.imshow(Inew1)
plt.show()
plt.waitforbuttonpress
