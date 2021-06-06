import onnxruntime
import cv2
import numpy as np
import os
from os import path
import argparse
from multiprocessing import Process, Queue

from pixmo.sort import Sort
from pixmo.detectors.yolo import (
    detect_objects,
    draw_labels,
    get_box_dimensions,
    load_yolo,
)

BASE_DIR = os.path.dirname(__file__)


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


def event_loop(queue: Queue):
    while True:
        msg = queue.get()
        if msg["state"] == "end":
            break
        print(f"{msg['state']}-{msg['payload']}")


def start_pixmo():
    event_queue = Queue(maxsize=50)
    event_process = Process(target=event_loop, args=((event_queue),))
    event_process.daemon = True
    event_process.start()

    object_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        model, classes, output_layers = load_yolo()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes = get_box_dimensions(outputs, height, width)
        track_bbs_ids = object_tracker.update(boxes)
        draw_labels(track_bbs_ids, frame, classes)
        key = cv2.waitKey(1)
        if key == 27:
            break
    event_queue.put({"state": "end"})
    event_process.join()
    cap.release()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="pixmo args")
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start_pixmo()
