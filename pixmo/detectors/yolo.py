from os import path
import cv2
import numpy as np

from pixmo.config import Config


def load_yolo():
    """
    Load yolo network through opencv: config + weights
    """
    net = cv2.dnn.readNet(
        path.join(Config.BASE_DIR, "models/yolov3.weights"),
        path.join(Config.BASE_DIR, "models/yolov3.cfg"),
    )
    classes = []
    with open(path.join(Config.BASE_DIR, "models/coco.txt"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_objects(img, net, outputLayers):
    """
    Preprocess the image
    pass it through the network
    Return detected outputs
    """
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
    height, width, channels = img.shape
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    """
    Process the image output to opencv version
    """
    boxes = []
    nms_boxes = []
    confs = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.4:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h, class_id])
                confs.append(float(conf))
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h, class_id = boxes[i]
            nms_boxes.append([x, y, x + w, y + h, confs[i], class_id])
    return np.array(nms_boxes) if len(nms_boxes) else np.empty((0, 6))


def draw_labels(boxes, img, classes):
    """
    render labels for the corresponding boxes
    """
    for box in boxes:
        (x1, y1, x2, y2, track_id, class_id) = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{track_id} - :{classes[class_id]}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            color,
            1,
        )
    cv2.imshow("Image", img)
