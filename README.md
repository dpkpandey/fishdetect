# FishCount

FishCount uses the YOLO model to count fish in images or videos.

## 1. Install Python
Ensure Python is installed on your computer. You can work in a local environment such as IDLE, Visual Studio, PyCharm, or use Google Colab or Jupyter Notebook. This guide focuses on a local setup, but instructions for Colab are also provided.

To install Python:
- **Windows**: Download and install the latest version from [Python's official website](https://www.python.org/downloads/).
- **Linux**: Open a terminal and run:
  ```bash
  sudo apt-get update && sudo apt-get upgrade
  sudo apt-get install python3 -y
  ```

## 2. Set Up Your Working Directory
Create a new folder for your project. Example:

- **Windows**:
  ```bash
  cd Desktop\fishdetect
  ```
- **Linux**:
  ```bash
  cd Desktop/fishdetect
  ```

## 3. Create a Virtual Environment
To manage dependencies, create and activate a virtual environment:

- **Windows**:
  ```bash
  python -m venv myenv
  myenv\Scripts\activate
  ```
- **Linux**:
  ```bash
  python3 -m venv myenv
  source myenv/bin/activate
  ```

To exit the virtual environment, type:
```bash
deactivate
```

## 4. Install Dependencies
Before proceeding, upgrade `pip` and install required libraries:
```bash
python3 -m pip install --upgrade pip
pip install ultralytics opencv-python numpy openpyxl torch
```

## 5. Enable GPU Acceleration (Optional)
For better performance with large datasets, configure GPU support:

1. Install the compatible CUDA version for your GPU. Check [Nvidia's website](https://developer.nvidia.com/cuda-downloads).
2. Install PyTorch with CUDA support from [PyTorch's official website](https://pytorch.org/).
3. Run the recommended command in your terminal to install PyTorch with CUDA.

## 6. Prepare Dataset
1. Sign up or log in to [Roboflow](https://roboflow.com/).
2. Upload and annotate images.
3. Download the dataset and organize files:
   - Place images in the `images` folder.
   - Place annotation `.txt` files in the `labels` folder inside your `fishdetect` directory.

## 7. Create YAML Configuration File
Create `fishdetect.yaml` to define the dataset path and classes:
```yaml
path: C:\Users\YourComputer\Desktop\fishdetect
train: images
val: images
names:
  0: fish
```
For multiple classes (e.g., algae detection):
```yaml
path: C:\Users\YourComputer\Desktop\algaedetect
train: images
val: images
nc: 15
names: ['Ceratium', 'Chaetoceros', 'Cyclotella', 'Cyanobacteria', 'Euglenoid Eutreptiella', 'Gymnodinium', 'Microcystis', 'New', 'Oocystis', 'Oocytis', 'Oscillatoria', 'Pleurosigma sp.', 'Pseudo-nitzschia', 'Pseudo-nitzschia sp.', 'macro-algae']
```

## 8. Train the Model
Create `main.py` with the following content:
```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.yaml").load("yolo11m.pt")  # Train from scratch
# model = YOLO("yolo11m.pt")  # Load pre-trained model (recommended)

# Train the model
results = model.train(data="fishdetect.yaml", batch=8, epochs=500)
```
- Adjust batch size according to your GPU memory.
- Run the script:
  ```bash
  python main.py
  ```

## 9. Run Detection
After training, see the run/train/weight folder and you will see last.pt and best.pt files, use those file and rename it like here I renamed " lastsmall1000all.pt"  and keep in your working directory name as detectfish.  
We use the model to detect fish in images or videos:
```python
import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("lastsmall1000all.pt")
results = model.predict("outputfile.mp4", show=True, save=True, conf=0.30, tracker ="botsort.yaml")
cv2.imshow("Detection", result.plot())
if cv2.waitKey(1) & OxFF ==ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
```

## 10. Count
Congratulations! You've successfully set up and trained a YOLO model to count fish.
Now, We need to count them, 
I have attached all the necessary documents except footage for footage you can download from "mydataset"
Here you can fine tune sort.py file such that you can easily get desirable results. In classes.txt, we 
define classes name. If you are using multiple classes just include in that and make a change in count.py as well.






For improvements or contributions, feel free to submit a pull request!

