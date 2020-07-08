DETECTING OBJECTS IN AN IMAGE USING OPENCV AND MOBILENET

In this project, MobileNet is used along with SSD (Single Shot Detection). 

MobileNet has a faster and lightweight network architecture as compared to ResNet and VGG.

SSD technique provides the accuracy as that of a Faster RCNN and the training speed as that of a YOLO (You Only Look Once).
(Faster R-CNN wasn't used, as training such a model takes quite a lot of time and YOLO wasn't used, as it is indeed good with the training speed but trades off with the accuracy.)

This pretrained model is fed with an input using blob. The input is then reshaped for the CNN and undergoes a forward pass into the net. The CNN outputs the class label along with the bounding box co-ordinates.

These co-ordicnates are then mapped onto the image using another code snippet in the program.


    Please find the caffe framework's model network ('MobileNetSSD_deploy.caffemodel') in the following link:
    https://drive.google.com/open?id=1aXUuTlKZBVXgvtO8lpiRFn1gBDDqgzxx

    Below are the classes names for which you can input the images of and try it yourself!
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"

Output preview:

![alt text](https://raw.githubusercontent.com/ShashankNardekar/ML_projects/master/CNN_projects/object_detection/example_03_output.PNG "Detection of horse and jockey")

*If you are not able to view this image, please find the same in the uploaded files with the name: example_03_output*
