class BehavoirEngine:
    def __init__(self):
        self.lock = False
        self.present_action = {}

    def update(self, msg: dict, emotion_engine):
        if msg["type"] == "internal_state_bad":
            return ("sad", None)

        payload = msg["payload"]
        frame = payload.get("frame")
        objects = payload.get("objects")

        if msg["type"] == "person":
            emotion = emotion_engine.update(frame)
            return ("person", emotion)
        else:
            print(f"Object {len(objects)}")
            return ("object", len(objects))
