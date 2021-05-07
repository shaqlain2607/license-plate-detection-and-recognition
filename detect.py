import os
import cv2
import numpy as np
from imageai.Detection.Custom import DetectionModelTrainer,CustomObjectDetection
from imageai.Detection.Custom import CustomVideoObjectDetection
import os


detector=CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('detection_model-ex-014--loss-0011.454.h5')
detector.setJsonPath('detection_config.json')
detector.loadModel()
detections=detector.detectObjectsFromImage(input_image='test_multipleCar/t (6).jpg',output_image_path='found.jpg', minimum_percentage_probability=35)
flag=1
img= cv2.imread('test_multipleCar/t (6).jpg', cv2.IMREAD_COLOR)
# img2 = cv2.imread('test13.jpg', cv2.IMREAD_COLOR)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
i=0
for detection in detections:
  if detection["name"]=='licence':
    print("found!")
    print(detection["name"] , " : ", detection["percentage_probability"], " : ", detection["box_points"] )
    # tl = (detection["box_points"][0],detection["box_points"][1])
    # br = (detection["box_points"][2],detection["box_points"][3])
    # # label= detection['name']
    # # percent=str(detection["percentage_probability"])
    # img = cv2.rectangle(img, tl, br, (0,255 , 0), 3)
    
    out = img[detection["box_points"][1]-2:detection["box_points"][3]+6, detection["box_points"][0]-2:detection["box_points"][2]+6]
    cv2.imshow('out'+str(i), out)
    cv2.waitKey(0)
    flag=0
    i=i+1
    cv2.imwrite('out'+str(i)+'.jpg', out)


if(flag):
  print('not found')
else:
    found= cv2.imread('found.jpg')
    found = cv2.resize(found, (min(900,found.shape[1]), min(500, found.shape[0])))
    cv2.imshow('fnd', found)
    cv2.waitKey(0)


# execution_path = os.getcwd()

# video_detector = CustomVideoObjectDetection()
# video_detector.setModelTypeAsYOLOv3()
# video_detector.setModelPath("detection_model-ex-014--loss-0011.454.h5")
# video_detector.setJsonPath("detection_config.json")
# video_detector.loadModel()
# video_detector.detectObjectsFromVideo(input_file_path="video.mp4",
#                                         output_file_path=os.path.join(execution_path, "video1"),
#                                         frames_per_second=140,
#                                         minimum_percentage_probability=40,
#                                         log_progress=True)

