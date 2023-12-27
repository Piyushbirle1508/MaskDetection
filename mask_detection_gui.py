import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

faceNet = cv2.dnn.readNet("face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.model")

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def show_frame():
    _, frame = vs.read()
    if _:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        
        video_label.imgtk = frame
        video_label.configure(image=frame)

        root.after(10, show_frame)

def open_file():
    global vs
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        vs = cv2.VideoCapture(file_path)
        show_frame()

root = tk.Tk()
root.title("Mask Detection App")

vs = None
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

open_button = tk.Button(root, text="Open Video File", command=open_file)
open_button.pack(pady=10)

quit_button = tk.Button(root, text="Quit", command=root.quit)
quit_button.pack(pady=10)

root.mainloop()