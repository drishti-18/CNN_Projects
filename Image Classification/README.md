
IMAGE CLASSIFIER USING GOOGLENET

An image classification project using OpenCV and GoogleNet (pretrained) using Caffe framework. There are around 1000 classes that this model can classify with quite a good accuracy. 

    Please find all the classes' names in the synset_words.txt

This pretrained model is fed with an input using blob. The input is then reshaped for the CNN and undergoes a forward pass into the net. The CNN outputs the class label. 

This code also provides the output label embedded in the image given as the input.

    Please find the pretrained model file (bvlc_googlenet.caffemodel) in this link: https://drive.google.com/open?id=1trBiq8PYX01M6QWDKK-G37EKdV4SMRty


