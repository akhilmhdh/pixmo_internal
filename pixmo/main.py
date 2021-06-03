import onnxruntime
import cv2
import numpy as np
import os
from os import path
import argparse
import torch
from pixmo.sort import Sort
from pixmo.utils.yolo import non_max_suppression_fast

BASE_DIR = os.path.dirname(__file__)


ort_session = onnxruntime.InferenceSession(path.join(BASE_DIR, "models/yolov5s.onnx"))

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


# def start_pixmo():
#     object_tracker = Sort()
#     face_cascade = cv2.CascadeClassifier(
#         path.join(BASE_DIR, "models/haarcascade_frontalface_default.xml")
#     )
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break

#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(image, 1.3, 5)
#         faces = (
#             np.array([[*face, 1] for face in faces]) if len(faces) else np.empty((0, 5))
#         )
#         track_bbs_ids = object_tracker.update(faces)
#         for i, (x, y, w, h, object_id) in enumerate(faces):
#             wpad = int(w * pad_scale)
#             hpad = int(h * pad_scale)
#             face_img = image[y - hpad : y + h + hpad, x - wpad : x + w + wpad]
#             [face_width, face_height] = face_img.shape
#             if face_width == 0 or face_height == 0:
#                 continue
#             face_img = cv2.resize(face_img, (48, 48))
#             face_img_tensor = img2tensor(face_img)[None][None]
#             ort_inputs = {ort_session.get_inputs()[0].name: face_img_tensor}
#             ort_outs = ort_session.run(None, ort_inputs)
#             output = ort_outs[0]
#             pred = np.argmax(output[0])
#             cv2.rectangle(frame, (x, y), (x + w, x + h), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 f"{classes[pred]} - person:{object_id}",
#                 (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 (36, 255, 12),
#                 2,
#             )
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) == ord("q"):
#             break
# cap.release()


def load_yolo():
    net = cv2.dnn.readNet(
        path.join(BASE_DIR, "models/yolov3.weights"),
        path.join(BASE_DIR, "models/yolov3.cfg"),
    )
    classes = []
    with open(path.join(BASE_DIR, "models/coco.txt"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=0.00392,
        size=(320, 320),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)


# def start_pixmo():
#     object_tracker = Sort()
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     while True:
#         ret, frame = cap.read()
#         model, classes, colors, output_layers = load_yolo()
#         height, width, channels = frame.shape
#         blob, outputs = detect_objects(frame, model, output_layers)
#         boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
#         draw_labels(boxes, confs, colors, class_ids, classes, frame)
#         key = cv2.waitKey(1)
#         if key == 27:
#             break
#     cap.release()


def start_pixmo():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        results = model(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="pixmo args")
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start_pixmo()
