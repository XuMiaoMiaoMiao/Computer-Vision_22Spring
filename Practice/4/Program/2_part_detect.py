import cv2
import matplotlib.pyplot as plt

I = cv2.imread('pictures/good_luck.jpg')
#I = cv2.imread('pictures/Frozen.png')
#I = cv2.imread('pictures/harry_potter.png')
Igray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

detector_face = cv2.CascadeClassifier()
cascade_face = cv2.samples.findFile("features/haarcascade_frontalface_default.xml")
detector_face.load(cascade_face)
detector_eye = cv2.CascadeClassifier()
cascade_eye = cv2.samples.findFile("features/haarcascade_eye.xml")
detector_eye.load(cascade_eye)
detector_mouth = cv2.CascadeClassifier()
cascade_mouth = cv2.samples.findFile("features/haarcascade_mcs_mouth.xml")
detector_mouth.load(cascade_mouth)
detector_nose = cv2.CascadeClassifier()
cascade_nose = cv2.samples.findFile("features/haarcascade_mcs_nose.xml")
detector_nose.load(cascade_nose)
faces = detector_face.detectMultiScale(Igray, scaleFactor = 1.07, minNeighbors = 12)
Iout = I.copy()
for (x,y,w,h) in faces:
    Iout = cv2.rectangle(Iout,(x,y,w,h),(0,255,255),3)
    Iface = Igray[y:y+h,x:x+w]
    Iface_top = Igray[y:y+h*2 // 3,x:x+w]
    Iface_bottom = Igray[y+(h//5*3):y+h,x:x+w]
    Iface_mid = Igray[y+(h//4):y+(h//4*3),x:x+w]
    eyes = detector_eye.detectMultiScale (Iface_top, scaleFactor = 1.05, minNeighbors = 1)
    mouths = detector_mouth.detectMultiScale (Iface_bottom, scaleFactor = 1.05, minNeighbors = 1)
    noses = detector_nose.detectMultiScale (Iface_mid, scaleFactor = 1.05, minNeighbors = 1)
    for (x2,y2,w2,h2)in eyes:
        Iout = cv2.rectangle(Iout,(x+x2,y+y2,w2,h2),(0,255,0),3)
    for (x3,y3,w3,h3)in mouths:
        Iout = cv2.rectangle(Iout,(x+x3,y+(h//5*3)+y3,w3,h3),(255,0,0),3)
    for (x4,y4,w4,h4)in noses:
        Iout = cv2.rectangle(Iout,(x+x4,y+(h//4)+y4,w4,h4),(0,0,255),3)
I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
Iout = cv2.cvtColor(Iout,cv2.COLOR_BGR2RGB)
plt.imshow(I),plt.title('original image with faces'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress

plt.imshow(Iout),plt.title(' Image with detected faces highlighted'),plt.axis ('off') 
plt.show()
plt.waitforbuttonpress

