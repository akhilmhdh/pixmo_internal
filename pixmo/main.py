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
from pixmo.emotion_engine import EmotionEngine

# https://arcade.academy/examples/happy_face.html

yolo_classes = []


def event_loop(queue: Queue):
    emotion_engine = EmotionEngine()
    while True:
        msg = queue.get()
        if msg["type"] == "end":
            break
        frame = msg["payload"]
        emotion = emotion_engine.update(frame)


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
