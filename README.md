# FishCount
Use YOLO model to count fish

First Now,
Install python in your computer. 
Where do you want to do this job. If you choose your local environment as IDLE or visual studio or any pycharm or Google Colab or Jupyternotebook.

I love to do in local environment rather than googlecolab but there is code for both. It is almost same]
Lets dig in 

This is where we need to mainly focus. As we are performing this work in python
so first of all we need to install python in our computer. It is better to install recent
version of the software. ( sometimes we might need to downgrade software version be-
cause of compatibilities of other module). If you are working in Windows just download
python and install it. If you are woking in Linux then, Go to terminal and type sudo
apt-get update && sudo apt-get upgrade && sudo apt-get install python3
-y .


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

python3 -m pip install –upgrade pip

pip install ultralytics

pip install opencv

pip install numpy

pip install openpyxl

pip install torch

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

Now, in fishdetect.yaml file it should be like this
Path: C:\ Users\ yourcomputer\ Desktop\ fishdetect
Train: images
Val: images
#Classes:
names:
  0: fish

This explains here the path of our training datasets, which folder to choose for training
and validation and what classes are available in our data. 

Well if you have more than one class then you can define here at first after annotation.
Like if we are working with algae then

path: C:\Users\Yourcomputer\Desktop\algaedetect
train: images
val: images
nc: 15
names: [‘Ceratium’, ‘Chaetoceros’, ‘Cyclotella’, ‘Cynobacteria’,
‘Euglenoid Eutreptiella’, ‘Gymnodinium’, ‘Microcyctis’, ‘New’,
‘Oocystis’, ‘Oocytis’, ‘Oscillatoria’, ‘Pleurosigma sp.’,
‘Pseudo-nitzchia’, ‘Pseudo-nitzschia sp.’, ‘macro-algae’]

After this lets make main.py file where we are going to train our data to get customize
pytorch model to detect fish. Now, in main.py

from ultralytics import YOLO
#load a model
model= YOLO("yolov11m.yaml").load("yolo11m.pt") # if you want to build from
a scratch use this command
#model = YOLO(" yolo11m.pt") #Load pre-trained model (recommended for training)
results= model.train(data ="fishdetect.yaml", batch= 8, epochs =500)

#choose batch size according to your GPU memory
Now go in above terminal and run main.py then you just need to wait until it finishes.
Well done.

