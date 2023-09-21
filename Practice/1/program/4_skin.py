import cv2
import matplotlib.pyplot as plt
from numpy import zeros_like
# uniform day light illumination
def uniform_day_skin(image):
    '''
    image : RGB image
    '''
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    m = image.shape[0]
    n = image.shape[1]
    skin = zeros_like(image)
    for i in range(m):
        for j in range(n):
            r = R[i,j]
            b = B[i,j]
            g = G[i,j]
            maxi = max(r,g,b)
            mini = min(r,g,b)
            if r>95 and g>40 and b>20 and maxi-mini>15 and abs(r-g)>15 and r>g and r>b:
                skin[i,j] = 255
    return skin
#under flash light or daylight lateral illumination
def flash_light_skin(image):
    '''
    image : RGB image
    '''
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    m = image.shape[0]
    n = image.shape[1]
    skin = zeros_like(image)
    for i in range(m):
        for j in range(n):
            r = R[i,j]
            b = B[i,j]
            g = G[i,j]
            if r>220 and g>210 and b>170 and  abs(r-g)<=15 and g>b and r>b:
                skin[i,j] = 255
    return skin 
#using normalized RGB values
def normalized_RGB(image):
    '''
    image : RGB image
    '''
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    m = image.shape[0]
    n = image.shape[1]
    skin = zeros_like(image)
    for i in range(m):
        for j in range(n):
            if (float(R[i,j])+float(G[i,j])+float(B[i,j])) != 0:
                r = float(R[i,j])/(float(R[i,j])+float(G[i,j])+float(B[i,j]))
                b = float(B[i,j])/(float(R[i,j])+float(G[i,j])+float(B[i,j]))
                g = float(G[i,j])/(float(R[i,j])+float(G[i,j])+float(B[i,j]))
                if g != 0 and r/g>1.185 and r*b/pow((r+g+b),2) >0.107 and r*g/pow((r+g+b),2)>0.112:
                    skin[i,j] = 255
    return skin  
#Uniform sunshine picture
I1 = cv2.imread("pictures/kid_face.jpg",cv2.IMREAD_COLOR)
I1= cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
#High exposure picture
I2 = cv2.imread("pictures/woman_face.jpg",cv2.IMREAD_COLOR)
I2= cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
# uniform day light illumination
uniform_1 = uniform_day_skin(I1)
uniform_2 = uniform_day_skin(I2)
#under flash light or daylight lateral illumination
flash_1 = flash_light_skin(I1)
flash_2 = flash_light_skin(I2)
#using normalized RGB values
normal_1 = normalized_RGB(I1)
normal_2 = normalized_RGB(I2)
#display image
plt.subplot(221),plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.imshow(I1)
plt.subplot(222),plt.title('uniform day light formula'),plt.xticks([]),plt.yticks([])
plt.imshow(uniform_1,cmap="gray")
plt.subplot(223),plt.title('under flash light or daylight formula'),plt.xticks([]),plt.yticks([])
plt.imshow(flash_1,cmap="gray")
plt.subplot(224),plt.title('normalized RGB values formula'),plt.xticks([]),plt.yticks([])
plt.imshow(normal_1,cmap="gray")
plt.show()
plt.waitforbuttonpress

plt.subplot(221),plt.title('original image'),plt.xticks([]),plt.yticks([])
plt.imshow(I2)
plt.subplot(222),plt.title('uniform day light formula'),plt.xticks([]),plt.yticks([])
plt.imshow(uniform_2,cmap="gray")
plt.subplot(223),plt.title('under flash light or daylight formula'),plt.xticks([]),plt.yticks([])
plt.imshow(flash_2,cmap="gray")
plt.subplot(224),plt.title('normalized RGB values formula'),plt.xticks([]),plt.yticks([])
plt.imshow(normal_2,cmap="gray")
plt.show()
plt.waitforbuttonpress
