FishCount

FishCount uses the YOLO model to count fish in images or videos.

1. Install Python

Ensure Python is installed on your computer. You can work in a local environment such as IDLE, Visual Studio, PyCharm, or use Google Colab or Jupyter Notebook. This guide focuses on a local setup, but instructions for Colab are also provided.

To install Python:

Windows: Download and install the latest version from Python's official website.

Linux: Open a terminal and run:

sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3 -y

2. Set Up Your Working Directory

Create a new folder for your project. Example:

Windows:

cd Desktop\fishdetect

Linux:

cd Desktop/fishdetect

3. Create a Virtual Environment

To manage dependencies, create and activate a virtual environment:

Windows:

python -m venv myenv
myenv\Scripts\activate

Linux:

python3 -m venv myenv
source myenv/bin/activate

To exit the virtual environment, type:

deactivate

4. Install Dependencies

Before proceeding, upgrade pip and install required libraries:

python3 -m pip install --upgrade pip
pip install ultralytics opencv-python numpy openpyxl torch

5. Enable GPU Acceleration (Optional)

For better performance with large datasets, configure GPU support:

Install the compatible CUDA version for your GPU. Check Nvidia's website.

Install PyTorch with CUDA support from PyTorch's official website.

Run the recommended command in your terminal to install PyTorch with CUDA.

6. Prepare Dataset

Sign up or log in to Roboflow.

Upload and annotate images.

Download the dataset and organize files:

Place images in the images folder.

Place annotation .txt files in the labels folder inside your fishdetect directory.

7. Create YAML Configuration File

Create fishdetect.yaml to define the dataset path and classes:

path: C:\Users\YourComputer\Desktop\fishdetect
train: images
val: images
names:
  0: fish

For multiple classes (e.g., algae detection):

path: C:\Users\YourComputer\Desktop\algaedetect
train: images
val: images
nc: 15
names: ['Ceratium', 'Chaetoceros', 'Cyclotella', 'Cyanobacteria', 'Euglenoid Eutreptiella', 'Gymnodinium', 'Microcystis', 'New', 'Oocystis', 'Oocytis', 'Oscillatoria', 'Pleurosigma sp.', 'Pseudo-nitzschia', 'Pseudo-nitzschia sp.', 'macro-algae']

8. Train the Model

Create main.py with the following content:

from ultralytics import YOLO

# Load a model
model = YOLO("yolov11m.yaml").load("yolo11m.pt")  # Train from scratch
# model = YOLO("yolo11m.pt")  # Load pre-trained model (recommended)

# Train the model
results = model.train(data="fishdetect.yaml", batch=8, epochs=500)

Adjust batch size according to your GPU memory.

Run the script:

python main.py

9. Run Detection

After training, use the model to detect fish in images or videos:

results = model.predict("test_image.jpg")

10. Conclusion

Congratulations! You've successfully set up and trained a YOLO model to count fish.

For improvements or contributions, feel free to submit a pull request!

