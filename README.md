# Computer-Vision_22Spring
Computer Vision projects during 22Spring semester at ITMO university

Below is a simple table of contents for the Practice : 

- **Practice 1** ： **<font color='cornflowerblue'>basic methods for images segmentation into semantic areas.</font>**
  - Binarization.**<font color='orange'>（upper and lower binarization thresholds）</font>**
  - Segmentation 1 
    - Image segmentation based on Weber principle
    - Segmentation of RGB images by skin color
  - Segmentation 2
    -  image segmentation in the CIE Lab color space by the <font color='cornflowerblue'>nearest neighbors method .</font>
    - image segmentation in the CIE Lab color space by the <font color='cornflowerblue'> 𝑘-means method .</font>
  - Segmentation 3
    - Texture segmentation using  mean value, Standard deviation,  relative smoothness, local entropy.
- **Practice 2** ： <font color='cornflowerblue'>**Hough Transform.**</font>
  - Search for lines.
    -  Search for straight lines using the Hough transform both for the original image and for the image obtained using differential operator.
  - Search for circles.
    - Search for circles of both a certain radius and from a given range using the Hough transform, both for the original image and for the image obtained using differential operator.
  - classic Hough transform algorithms for lines,  Highlight the selected points in the Hough parameter space.
  - classic Hough transform algorithms for circles,  Highlight the selected points in the Hough parameter space.
  - Compare implementation results.
- **Practice 3** ： <font color='cornflowerblue'>**Features Detectors.**</font>
  - Feature points detection.
    - Using SIFT feature point descriptor
    - Using ORB feature point descriptor
  - Feature points matching. 
    - <font color='cornflowerblue'>Extract feature points</font> of an object and <font color='cornflowerblue'>match</font> them with feature points of a scene containing this object, Calculate the transformation matrix using **RANSAC method** and highlight the object position in the scene.
    - Compare feature point descriptors for the task of image matching.
  - simple automatic image stitching.
    - calculate the transformation matrix between two images and stitch them into a single panoramic image.
    - stitch three images into a single panoramic image.
- **Practice 4** ： <font color='cornflowerblue'>**Face Detection using Viola-Jones Approach.**</font>
  - Faces detection.
    - search faces using <font color='cornflowerblue'>Viola-Jones approach.</font>
  - Body parts detectiong.
    -  Search for eyes, mouth, and nose in one image, and use ROI to improve accuracy.
  - face detection in videostream using pre-recorded video with faces.
    - [★bilibili video : Facial Recognition of Fragments from ”Goodluck Charlie ”(Viola-Jones Approach)★](https://www.bilibili.com/video/BV1Wg411R7Eq/?spm_id_from=333.999.0.0)
  - face detection in live videostream using web-camera.
