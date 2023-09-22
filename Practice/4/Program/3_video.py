import cv2
from moviepy import *
from moviepy.editor import *
def op_one_img(I):
    Igray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier()
    cascade_fn = cv2.samples.findFile("features\haarcascade_frontalface_default.xml")
    detector.load(cascade_fn)
    faces = detector.detectMultiScale(Igray, scaleFactor = 1.07, minNeighbors = 15)
    Iout = I.copy()
    for (x,y,w,h) in faces:
        Iout = cv2.rectangle(Iout,(x,y,w,h),(0,255,255),3)
    return Iout


def makevideo():
    videoinpath  = 'video\original_video.avi'
    videooutpath = 'video\detect_out.avi'
    capture     = cv2.VideoCapture(videoinpath  )
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc      = cv2.VideoWriter_fourcc(*'XVID')
    writer      = cv2.VideoWriter(videooutpath ,fourcc, fps, (1024,576), True)
    if capture.isOpened():
        while True:
            ret,img_src=capture.read()
            if not ret:break
            img_out = op_one_img(img_src)
            # Frame-by-frame processing
            writer.write(img_out)
    else:
        print('Video opening failed!')
    writer.release()

makevideo()

video = VideoFileClip('video\original_video.avi')
video_out = VideoFileClip('video\detect_out.avi')
audio = video.audio
#audio.write_audiofile('audio.mp3')
videoclip = video_out.set_audio(audio)
videoclip.write_videofile('result.mp4')

