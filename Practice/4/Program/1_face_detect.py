import cv2
import matplotlib.pyplot as plt

I = cv2.imread('pictures/good_luck.jpg')
#I = cv2.imread('pictures/Frozen.png')
#I = cv2.imread('pictures/harry_potter.png')
Igray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

detector = cv2.CascadeClassifier()
cascade_fn = cv2.samples.findFile("features/haarcascade_frontalface_default.xml")
detector.load(cascade_fn)
faces = detector.detectMultiScale(Igray, scaleFactor = 1.07, minNeighbors = 12)
Iout = I.copy()
for (x,y,w,h) in faces:
    Iout = cv2.rectangle(Iout,(x,y,w,h),(0,255,255),3)

I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
Iout = cv2.cvtColor(Iout,cv2.COLOR_BGR2RGB)
plt.imshow(I),plt.title('original image with faces'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress

plt.imshow(Iout),plt.title(' Image with detected faces highlighted'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress

