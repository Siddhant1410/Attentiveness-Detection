import numpy as np
import cv2
import time
import os
import sys

x=1


while(True):
    # Capture frame-by-frame
    cap = cv2.VideoCapture("video3.mp4")
    framerate = cap.get(5)
    ret, frame = cap.read()
    cap.release()
    # Our operations on the frame come here
    filename = "frames/image_rend" + str(int(x)) + ".jpg"
    x=x+1
    cv2.imwrite(filename, frame)
    print("Frame Written Successfully!")
    time.sleep(5)
    if x == 8:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

os.system('python head_pose_estimation.py frames/image_rend2.jpg')
os.system('python head_pose_estimation.py frames/image_rend3.jpg')
os.system('python head_pose_estimation.py frames/image_rend4.jpg')
os.system('python head_pose_estimation.py frames/image_rend5.jpg')
os.system('python head_pose_estimation.py frames/image_rend6.jpg')
os.system('python head_pose_estimation.py frames/image_rend7.jpg')


os.remove("frames/image_rend1.jpg")
os.remove("frames/image_rend2.jpg")
os.remove("frames/image_rend3.jpg")
os.remove("frames/image_rend4.jpg")
os.remove("frames/image_rend5.jpg")
os.remove("frames/image_rend6.jpg")
os.remove("frames/image_rend7.jpg")
