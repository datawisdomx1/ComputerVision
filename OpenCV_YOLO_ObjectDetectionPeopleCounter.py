#!/usr/bin/env python
# coding: utf-8


import cv2
import torch
from PIL import Image

# Install the set of libraries as given in the requirements file to use ultralytics yolov5 model with pytorch
# pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

i = 0

while True:
    # Capture image from camera frame by frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't capture frame from stream. Exit...")
        break

    # Save each frame to drive using imwrite method
    cv2.imwrite('/PersonImage'+str(i)+'.jpg', frame)
    
    # Images
    im1 = Image.open('/PersonImage'+str(i)+'.jpg')  # PIL image
    imgs = [im1]  # batch of images

    # Inference
    results = model(imgs, size=640)  # includes NMS

    # Results
    results.print()
    results.show()
    
    # im1 predictions (pandas)
    j = 0
    for k in range(len(results.pandas().xyxy[0])):
        if(results.pandas().xyxy[0]['name'][k] == 'person'):
            j += 1

    i += 1
    
    print('Frame: %s, Total no of people: %s'%(i,j))
    
    if i==5 :
        break

# Release the camera capture object after exiting
cap.release()
cv2.destroyAllWindows()


# In[ ]:




