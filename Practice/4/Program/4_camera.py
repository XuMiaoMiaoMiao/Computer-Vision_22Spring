import cv2
face_detector = cv2.CascadeClassifier()
cascade_fn = cv2.samples.findFile('features\haarcascade_frontalface_alt.xml')
face_detector.load(cascade_fn)
# Get camera behavior
cap = cv2.VideoCapture(0)
while True:
    # Returns pictures by frame from the camera
    flag,frame = cap.read()
    if not flag : 
        break
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray, scaleFactor = 1.07, minNeighbors = 12)
    for (x,y,w,h) in faces:
        cat = cv2.imread('pictures\cat.jpg')
        cat = cv2.resize(cat,dsize = (w,h))
        frame[y:y+h,x:x+w] = cat
    cv2.imshow('face detection and replacement with cat',frame)
    key = cv2.waitKey(10)
    if key == ord('q'): # Enter q to quit reading
            break
cv2.destroyAllWindows()
cap.release()