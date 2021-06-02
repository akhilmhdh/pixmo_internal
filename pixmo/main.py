import onnxruntime
import cv2
import numpy as np
import os
from os import path
import argparse

BASE_DIR = os.path.dirname(__file__)


ort_session = onnxruntime.InferenceSession(path.join(BASE_DIR, "models/emotion.onnx"))

pad_scale = 0.33

classes = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def img2tensor(img):
    img = img / 255
    # Normalize the image to mean and std
    mean = [0.5]
    std = [0.5]
    img = (img - mean) / std
    img = np.array(img, dtype=np.float32)
    return img


def start_pixmo():
    image = cv2.imread(path.join(BASE_DIR, "test/dhoni.jpeg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        path.join(BASE_DIR, "models/haarcascade_frontalface_default.xml")
    )
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for i, (x, y, w, h) in enumerate(faces):
        wpad = int(w * pad_scale)
        hpad = int(h * pad_scale)
        face_img = image[y - hpad : y + h + hpad, x - wpad : x + w + wpad]
        face_img = cv2.resize(face_img, (48, 48))
        face_img_tensor = img2tensor(face_img)[None][None]
        print(face_img_tensor.shape)
        ort_inputs = {ort_session.get_inputs()[0].name: face_img_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        pred = np.argmax(output[0])
        print(classes[pred])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="pixmo args")
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start_pixmo()
