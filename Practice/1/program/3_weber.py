import cv2
import matplotlib.pyplot as plt
from numpy import zeros_like
import math

def Weber(I):
    if I>=0 and I<=88:
        W = 20 - 12*I/88
    elif I>88 and I<=138:
        W = 0.002*pow((I-88),2)
    else:
        W = 7*(I-138)/117 + 13
    return W

def Weber_segmentation(image):
    '''
    image : Input grayscale image
    '''
    m = image.shape[0]
    n = image.shape[1]
    W = zeros_like(image)
    Igraynew = zeros_like(image)
    for i in range(m):
        for j in range(n):
            W[i,j] = Weber(Igray[i,j])
    lower_bound = 0
    upper_bound = 0 + Weber(lower_bound)
    p = 0
    class1 = 1
    while p <m*n:
        print(p)
        for i in range(m):
            for j in range(n):
                if Igray[i,j]>=lower_bound and Igray[i,j]<=upper_bound:
                    Igraynew[i,j] = lower_bound
                    p = p+1
        lower_bound = math.floor(upper_bound + 1)
        upper_bound = lower_bound + Weber(lower_bound)
        class1 = class1 + 1
    return Igraynew,class1

#############################################################################
########main################
#input the image and transform it into grayscale image
I = cv2.imread("pictures/face.jpg",cv2.IMREAD_COLOR)
I= cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
#Weber segmentation
Igraynew,class1 = Weber_segmentation(Igray)
#display image
plt.subplot(121),plt.title('original grayscale image'),plt.xticks([]),plt.yticks([])
plt.imshow(Igray,cmap="gray")
plt.subplot(122),plt.title('Image after Weber segmentation'),plt.xticks([]),plt.yticks([])
plt.imshow(Igraynew,cmap="gray")
plt.show()
plt.waitforbuttonpress
print(class1)

