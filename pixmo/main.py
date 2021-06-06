import onnxruntime
import cv2
import numpy as np
import os
from os import path
import argparse
from multiprocessing import Process, Queue
import face_recognition
import onnxruntime

from pixmo.config import Config

from pixmo.sort import Sort
from pixmo.detectors.yolo import (
    detect_objects,
    draw_labels,
    get_box_dimensions,
    load_yolo,
)

ort_session = onnxruntime.InferenceSession(
    path.join(Config.BASE_DIR, "models/emotion.onnx")
)

# https://arcade.academy/examples/happy_face.html

yolo_classes = []
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


def event_loop(queue: Queue):
    while True:
        msg = queue.get()
        if msg["type"] == "end":
            break
        frame = msg["payload"]
        height, width, channels = frame.shape
        input_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(input_frame)
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            face_img = frame[top:bottom, left:right]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            face_img_tensor = img2tensor(face_img)[None][None]
            ort_inputs = {ort_session.get_inputs()[0].name: face_img_tensor}
            ort_outs = ort_session.run(None, ort_inputs)
            output = ort_outs[0]
            pred = np.argmax(output[0])

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame,
                f"akhil:{classes[pred]}",
                (left + 6, bottom - 6),
                font,
                1.0,
                (255, 255, 255),
                1,
            )
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


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
        model, output_layers = load_yolo()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, hasPerson = get_box_dimensions(outputs, height, width)
        if hasPerson:
            event_queue.put({"type": "person", "payload": frame})
        track_bbs_ids = object_tracker.update(boxes)
        draw_labels(track_bbs_ids, frame, yolo_classes)
        key = cv2.waitKey(1)
        if key == 27:
            break
    # Clean up
    event_queue.put({"type": "end"})
    event_process.join()
    cap.release()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="pixmo args")
    # parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    with open(path.join(Config.BASE_DIR, "models/coco.txt"), "r") as f:
        yolo_classes = [line.strip() for line in f.readlines()]
    start_pixmo()
