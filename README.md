# License-plate-detection-and-recognition
* Autonomous number plate recognition is a special form of optical character recognition (OCR).
It enable computer system to read automatically the license number of vehicles from digital pictures.
Reading automatically mean transforming the pixel of the digital  image into ASCII text of number plate.
Capturing of fast moving vehicles need special technique to avoid motion blur.

## Approach
* The problem is divided into three sub problems. These are-
1. Number plate extraction 
2. Character segmentation 
3. Character recognition

## Number plate extraction
* YOLO algorithm was used for the object detection to detect our target object – ‘license’. The imageai implementation of the yolov3 convolutional neural network was used for this.Imageai is a powerful python library which provides computer vision capabilities using deep learning. Before training anchor boxes were created
With an iou (intersection over union) of 0.74. Transfer learning was used as it gave better result than training from scratch.





 



