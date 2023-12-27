### MaskDetection ###

### Mask Detection System ###
### Introduction ###
```This project focuses on the detection of whether a person is wearing a mask or not using deep learning techniques. The system utilizes a Convolutional Neural Network (CNN) model implemented with TensorFlow and Keras for accurate mask detection. The project includes different components, such as a mask detection model, image detection script, video detection script, and a graphical user interface for real-time mask detection.```

### Project Structure ###
1) mask_detector.ipynb: Jupyter Notebook containing the code for training the mask detection model using a MobileNetV2 architecture.
2) detect_mask_image.py: Python script for detecting masks in a single image.
3) mask_detector_video.py: Python script for real-time mask detection in a video stream.
4) mask_detection_gui.py: Python script for a graphical user interface (GUI) application for mask detection from video files.

### Prerequisites ###
```Before using the Mask Detection System, make sure you have the following prerequisites installed:```

```Python 3.10```

```Required Python packages (you can install them using pip install -r requirements.txt)```

### Getting Started ###
### Training the Model
1) Gather your dataset and organize it into two categories: "with_mask" and "without_mask."
2) Place the images in the corresponding directories within the dataset folder.
3) Open and run the mask_detector.ipynb Jupyter Notebook to train the model. You can modify the parameters like initial learning rate, number of epochs, and batch size as needed.

### Detecting Masks in Images ###
1) Use the detect_mask_image.py script to detect masks in a single image.
```python detect_mask_image.py```

### Detecting Masks in a live Video Stream ###
1) Prepare your video stream (e.g., using a webcam or video file).
2) Run the mask_detector_video.py script to detect masks in the video stream.

### GUI Application ###
```Execute the mask_detection_gui.py script to launch the GUI application for mask detection from downloaded video files.```
```python mask_detection_gui.py```

### Dependencies ###
```Make sure you have the following Python packages installed, as specified in the requirements.txt file:```

1) numpy
2) pandas
3) seaborn
4) Werkzeug>=2.1.2
5) Flask>=2.1.2
6) tqdm>=4.64.1
7) tensorflow>=2.10.0
8) opencv_python
9) matplotlib
10) scikit_learn
11) passlib>=1.7.4
12) gunicorn==20.1.0
13) Jinja2
14) itsdangerous==2.1.0
15) MarkupSafe==2.1.1
16) imutils

### License ###
```This project is licensed under the MIT License. See the LICENSE file for details.```