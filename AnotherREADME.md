 # Fish Counting 
 If you want to dig little bit more in the theory behind this algorithms and how it is really working. In this section I am bit trying to explain, there could be some misleading
 so, anyone can correct if they find it wrong. YOU ARE MOST WELCOME TO CONTRIBUTE.
 As we are using, YOLO model to first detect and count the fish. In future we will do more than that. 
 I assume you have gone through $\textbf{\textcolor{purple}{README.md}}$ file. So, that I do not have to explain steps here.

## What is Machine Learning ?

 # YOLO (You Only Look Once) - Object Detection

YOLO is a real-time object detection algorithm that processes an image in a single pass through a neural network. Unlike traditional object detection methods, which apply classifiers to different regions of an image, YOLO treats detection as a single regression problem, predicting bounding boxes and class probabilities simultaneously.

## **How YOLO Works:**

1. **Input Image Splitting:**  
   - The input image is divided into an \( S \times S \) grid (e.g., 13×13 for YOLOv3 at 416×416 resolution).
   - Each grid cell is responsible for detecting objects whose center falls within it.

2. **Bounding Box Predictions:**  
   - Each grid cell predicts a fixed number of bounding boxes (B), typically 2–5.
   - Each bounding box includes:
     - \( x, y \) (coordinates relative to the grid cell)
     - \( w, h \) (width and height relative to the image)
     - Confidence score (probability of an object in the box × IoU with the ground truth box).

3. **Class Predictions:**  
   - Each grid cell also predicts class probabilities for detected objects.
   - The final score for each bounding box is **confidence × class probability**.

4. **Non-Maximum Suppression (NMS):**  
   - Since multiple boxes may detect the same object, NMS filters out redundant predictions.
   - The box with the highest confidence is kept, and overlapping boxes (IoU > threshold) are removed.

## **YOLO Versions:**

- **YOLOv1 (2015):** Introduced single-shot detection with real-time performance.
- **YOLOv2 (2016):** Improved accuracy and speed with batch normalization and anchor boxes.
- **YOLOv3 (2018):** Added multi-scale predictions and Darknet-53 backbone.
- **YOLOv4 (2020):** Optimized speed and accuracy with CSPDarknet.
- **YOLOv5 (2020, Ultralytics):** Not an official continuation but widely used, optimized for PyTorch.
- **YOLOv6, YOLOv7, YOLOv8:** Further refinements in efficiency and accuracy.

## **Why YOLO?**
✅ **Fast** – Real-time processing (~30–150 FPS).  
✅ **Accurate** – Good balance of speed and precision.  
✅ **End-to-End Learning** – Entire image processed in one forward pass.

Would you like help implementing YOLO for a specific task? 🚀

 ## HOW different tracks are important and how to use them?
 ## How Length and Weight of fish are calculated?
 ### How we save the file in computer especially export in excel file 
 ##  1. Install python in your computer. 
Where do you want to do this job. If you choose your local environment as IDLE or visual studio or any pycharm or Google Colab or Jupyternotebook.

I love to do in local environment rather than googlecolab but there is code for both. It is almost same]
Lets dig in 

This is where we need to mainly focus. As we are performing this work in python
so first of all we need to install python in our computer. It is better to install latest
version of the software. ( sometimes we might need to downgrade software version be-
cause of compatibilities of other module). If you are working in Windows just download
python and install it. If you are woking in Linux then, Go to terminal and type

```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3 -y
```


# 2. Set Up Your Working Directory
Create a new folder for your project. For example, create a folder named fishdetect on your Desktop. Navigate to this folder using the command prompt or terminal:
# Windows
cd Desktop\fishdetect

# Linux
cd Desktop/fishdetect

Now, you have python installed in your system. Find out the working directory where
you are going to work. I worked in both Ubuntu and Windows environment. So, I
made new folder in Desktop for my comfort and named as ”fishdetect” in both com-
puter. So, I will explain what to do. Go to command prompt or terminal according
to you Operating System (OS) then type cd Desktop/fishdetect ( For Windows use
\and for linux use /)


Then you will be on the fishdetect folder now you all need to do is to create virtual
local python environment just by typing

In Windows: python –m venv myenv

Then type myenv\Scripts\activate

Then python detect.py (this detect.py file is to run the command skip for now)

In Linux: python3 -m venv myenv

Then type source myenv/bin/activate

Then python3 detect.py


We are intend to work in Ultralytics, such that we need to install some packages before
we do other stuff.


Now lets do first upgrade our pip packages
```bash
python3 -m pip install –upgrade pip

pip install ultralytics

pip install opencv

pip install numpy

pip install openpyxl

pip install torch
```

As other packages as required. If you intended to get out from virtual environment the
just type deactivate.

For small work with less dataset you might be able to get everthing with CPU, for
large amount of data and work in real time we need to use GPU.

Enable GPU Accelerator for this simulation:
To access GPU capabilities, we need to focus on the compatibility of the cuda version
with the computer we are working on. For example, GEFORCE RTX 3060, cuda 12.6
pre version is compatible. So, first go to the Nvidia website for cuda and download
recommended version and install it on the computer. You might need admin access to
install this software. It will help you to use GPU for the pytorch. Go to this website
“https://pytorch.org/” and choose which pytorch build you want
to use. For lower version of the cuda compute platform choose your build. Then choose
your OS system. In this case I am using a pip so I will go for the pip package. If you
are working with annoconda you can choose conda package. And language is python
for this program. If you are working with C++ or Java, you can choose that and your
installed cuda version. Then you will get run command at bottom, copy and paste in
the terminal where you have installed virtual environment. Now you should be able to
run GPU enabled pytorch for simulation.
Now, before we jump in any we need data, lets jump in roboflow and sign up or log in.
For public project you get it for free thanks to them. Now insert images and annotate
and export them. Put all images in images folder and all annotated .txt file in labels
folder in your working directory, i.e., fishdetect. Now, we need to create .yaml file.
You can name as you want I am naming as fishdetect.yaml for my conveniences.


