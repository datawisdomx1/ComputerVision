# ComputerVision
Computer Vision examples - classification, object detection, transfer learning 
CNN, RNN, LSTM, OpenCv, YOLO
on CIFAR10, MNIST, COCO, other datasets

ComputerVision_PeopleCounter_OpenCV_YOLO5_Pytorch
Computer Vision model for livestream frame people counter. Using OpenCV, YOLO5 from ultralytics, pytorch YOLO5 used the COCO image dataset for training its model
Logic: The OpenCV library is used to get the livestream of images from the camera. The livestream is passed frame by frame to the YOLOv5 model for object detection. If the image contains a ‘person’, the people counter is updated. Total number of people in the image is printed at the end.
Results: The model performs well in controlled environment, with close range images, proper lighting, etc. 100% accuracy. The algorithm was able to detect the correct number of people in the image, even if partly visible and contain other objects.
The model performs poorly in an outside external environment. Accuracy drops below 50% For the close range images, the correct number of people are detected. For the long range images, the model does not detect people or objects in the image, which is a limitation of the camera quality (macbook air) and YOLOv5 model library being used. Model detects unique people when there is a major overlap of people in the frame. If people are beyond 20-25 meters or on higher floors, the model fails to detect people, as the cameras are located within the range and on the same floor as people we are trying to detect.
Summary: Overall its a good model and easy to use with pytorch. Using better camera or other models like SSD can improve the overall accuracy
The code can be extended to perform object detection on a variety of object categories (80 in COCO) and use the livestream instead of just frames. Links below for libraries and installation
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html https://pytorch.org/hub/ultralytics_yolov5/ https://ultralytics.com/ https://cocodataset.org/#home https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/ https://blog.roboflow.com/m1-opencv/

ComputerVision_CNN_CIFAR10_ImageClassification_Tensorflow
Using Tensorflow and keras for building CNN model with CIFAR-10 dataset for image classification
Built different CNN models using different techniques and layers Achieved an accuracy of more than 85% between train and test sets Final model gave an accuracy of 85% Used 3 convolution layers with MaxPooling, 3 Flattening Dense layers, ‘Adam’ optimizer, ‘relu’ and ‘LeakyReLU ‘ as activation functions BatchNormalization and Dropout were used in different layers Model was run with batch size =10 and epochs = 100 Model stabilized by 20 epochs (validation set) with reducing marginal improvement Model can be improved by using more convolution layers, different optimizers, activation functions, epochs and transfer learning. But that will require more compute.
You can try using different image datasets along with transfer learning to improve the model. Rest is all about hypertuning NN layers and their parameters.

