import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def detect_mask_image(image_path):
    # Set the paths for the face detector and mask detector model
    face_detector_path = "face_detector"
    mask_detector_path = "mask_detector.model"

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([face_detector_path, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model(mask_detector_path)

    # load the input image from disk, clone it, and grab the image spatial dimensions
    image = cv2.imread(image_path)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:  # You can adjust the confidence threshold as needed
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

def upload_image():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                           filetypes=(("Image files", "*.jpg; *.jpeg; *.png"), ("all files", "*.*")))
    if file_path:
        # Display the selected image
        image = Image.open(file_path)
        image = ImageTk.PhotoImage(image)
        panel = tk.Label(window, image=image)
        panel.image = image
        panel.grid(row=1, column=0, padx=10, pady=10)

        # Perform mask detection on the uploaded image
        detect_mask_image(file_path)

# Create the main window
window = tk.Tk()
window.title("Mask Detection App")

# Create and set window dimensions
window.geometry("600x400")

# Create and set window background color
window.configure(bg='#ffffff')

# Create and set window resizable
window.resizable(False, False)

# Create a button for image upload
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=10, pady=10)

# Run the Tkinter event loop
window.mainloop()