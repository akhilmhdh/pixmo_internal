import numpy as np


class BehavoirEngine:
    def __init__(self):
        self.lock = False
        self.present_action = {}

    def compute_largest_box(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        largest_area_index = np.argmax(area)
        return boxes[largest_area_index]

    def update(self, msg: dict, emotion_engine):
        if msg["type"] == "internal_state_bad":
            return ("sad", None)

        payload = msg["payload"]
        frame = payload.get("frame")
        objects = payload.get("objects")

        height, width, _ = frame.shape
        frame_cx, frame_cy = width // 2, height // 2

        if msg["type"] == "person":
            emotion = emotion_engine.update(frame)
            if (emotion != "neutral") and (emotion != "happy"):
                return ("person", emotion)

            if objects.shape[0] > 1:
                objects = objects[objects[:, 5] != 0]

        focused_object = self.compute_largest_box(objects)
        (x1, y1, x2, y2, track_id, class_id) = focused_object

        object_width, object_height = x2 - x1, y2 - y1
        cx, cy = x1 + (object_width // 2), y1 + (object_height // 2)
        left_axis = ((frame_cx - cx) / width) * 100
        right_axis = ((frame_cy - cy) / height) * 100
        # print(f"FW {frame_cx}: FH{frame_cy}: CW: {cx}: CH{cy}")
        pos = f"{int(left_axis)}:{int(right_axis)}"
        return ("look", pos)
