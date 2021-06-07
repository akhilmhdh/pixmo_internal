from datetime import datetime
from os import path
from collections import Counter
import cv2
import numpy as np
import face_recognition
import onnxruntime

from pixmo.config import Config

ort_session = onnxruntime.InferenceSession(
    path.join(Config.BASE_DIR, "models/emotion.onnx")
)

owner_image = face_recognition.load_image_file(
    path.join(Config.BASE_DIR, "faces/owner.jpg")
)
owner_image_encoding = [face_recognition.face_encodings(owner_image)[0]]


def img2tensor(img):
    img = img / 255
    # Normalize the image to mean and std
    mean = [0.5]
    std = [0.5]
    img = (img - mean) / std
    img = np.array(img, dtype=np.float32)
    return img


class EmotionEngine:
    def __init__(self, frame_rate=1, scoring_rate=5):
        self.start_interval = datetime.now()
        self.frame_rate = frame_rate
        self.scoring_rate = scoring_rate
        self.emotions = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }
        self.emotion_collector = []
        self.state = self.emotions[6]

    def update(self, frame):
        height, width, channels = frame.shape
        input_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(input_frame)
        face_encodings = face_recognition.face_encodings(input_frame, face_locations)
        face_names = []

        face_index = 0

        for encoding_index, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                owner_image_encoding, face_encoding
            )
            name = "Unknown"

            if True in matches:
                face_index = encoding_index
                name = "owner"

        top, right, bottom, left = face_locations[face_index]
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

        present_time = datetime.now()
        diff = present_time - self.start_interval
        if diff.total_seconds() > self.scoring_rate:
            self.start_interval = datetime.now()
            occurence_count = Counter(self.emotion_collector)
            emotion = occurence_count.most_common(1)[0][0]
            self.state = emotion
            self.emotion_collector = []
            return emotion
        else:
            self.emotion_collector.append(self.emotions[pred])
            return self.state
        # # Draw a box around the face
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # # Draw a label with a name below the face
        # cv2.rectangle(
        #     frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        # )
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(
        #     frame,
        #     f"{face_names[index]}:{classes[pred]}",
        #     (left + 6, bottom - 6),
        #     font,
        #     1.0,
        #     (255, 255, 255),
        #     1,
        # )

    # cv2.imshow("frame", frame)
    # key = cv2.waitKey(1)
