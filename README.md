# Realtime Video Object Detector
This project is about object detection in real time video files. Computer vision is a huge part of the data science/AI domain which is substantially advanced over the last couple of years.

## Object Detection Task

The object detection task is based on following two prerequisite stages:

- **Image classification:** a supervised learning process to train an algorithm to recognize a set of target classes (objects to identify in images) using labeled example photos.
- **Image classification with localization:** the process of training a supervised algorithm to predict class as well as the bounding box around the object in the image. The term 'localization' refers to identifying the location of the object in the image. 

The **object detection**  task refers to the situation where the model needs to detect multiple objects in the picture and localized them all by defining a bounding box around them. A prime example is an autonomous driving application, which needs to detect not just other cars, but also other pedestrians and motorcycles and even other objects. 


## YOLO algorithm
In recent years, deep learning algorithms are offering cutting-edge improved results for object detection. YOLO algorithm is one of the most popular Convolutional Neural Networks with a single end-to-end model that can perform object detection in real-time. YOLO stands for, You Only Look Once and is an algorithm developed by Joseph Redmon, et al. and first described in the 2015 paper titled [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

The creation of the algorithm stemmed from the idea to place down a grid on the image and apply the image classification and localization algorithm to each of the grids.
Here the YOLOv3, a refined design which uses predefined anchor boxes to improve bounding box, is utilized for object detection in new images. Source code and pre-trained models of YOLOv3 is available in the [official DarkNet website](https://pjreddie.com/darknet/yolo/).

### Tiny-YOLOv3
Tiny YOLOv3 is a simplified version of YOLOv3. It is a fast real-time detection algorithm developed for embedded devices. The model structure is simple and the accuracy is not as high as the full version. A higher miss detection rate is expected for small targets. This model was trained with the Coco data set and can detect up to 80 classes. 

## OpenCV
OpenCV is an open source library which provides tools to perform image and video processing for the computer vision, machine learning, and image processing applications. It is particularly popular for real-time operations which is very important in today’s systems. Integration with various libraries, such as Numpuy and python resulted in great capablities of processing the OpenCV array structures for analysis and mathematical operations.

## Project background and objectives

This project is a collection of developed utility functions to facilitate usage of YOLOv3-tiny for object detection in real time video files. The developed functions achieved the following objectives :

* download the YOLOv3-tiny pre-trained model weights
* define a OpenCV DNN module and load the downloaded model weights
* create a cv2.VideoCapture object to load video files and read it frame-by-frame 
* perform object detection process on each frame of the video file (similar to what was described in my previous project <a href="https://github.com/majid-hosseini/Image-Object-Detector" target="_blank">**Image Object Detector**</a>)
* run the model and mark detected objects in each frame of video file
	*plotted bounding boxed and the class names around each detected object 
* processed frames are displayed using cv2.imshow method
* save the video of processed frames indicating detected objects
*close video files and destroy the window, which was created by the imshow method using cap.release() and cv2.destroyAllWindows() commands. 

## Sample ouput
Following is a sample output of this project showing detected objects along with the class label on each frame of a traffic video. You cal also find it in the folder "/videos" of this project

![Sample-output](https://github.com/majid-hosseini/Realtime-Object-Detector/blob/main/videos/Sample_output.gif)

 


# Requirement
* OpenCV 4.2.0
* Python 3.6


# Quick start
* **Download official** <a href="https://pjreddie.com/media/files/yolov3-tiny.weights" target="_blank">**YOLOv3-tiny Pre-trained Model Weights**</a> **and place it in the same folder of project**



## Dependencies
* OpenCV
* Numpy


How to use?
===========
The project is developed in Jupyter Notebooks which is automatically rendered by GitHub. The developed codes and functions for each step are explained in the notebook.




