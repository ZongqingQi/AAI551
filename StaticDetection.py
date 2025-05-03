import os
import cv2
import json
import torch

'''
This class is used to perform object recognition on static images, and the recognition results will be saved in a new directory.
'''

class StaticDetection:
    def __init__(self, conf=0.75, threshold=1, target_type="person"):
        self.conf = conf
        self.threshold = threshold
        self.target_type = target_type

        # Map the tag with the keys in model results
        self.target_map = {
            "person": ["person"],
            "pets": ["cat", "dog"],
            "vehicles": ["car", "truck", "bus", "motorcycle"],
            "airplane": ["airplane"]
        }

        # load YOLOv5 Model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = self.conf

    def detect(self, img_dir, output_dir, info_file):
        os.makedirs(output_dir, exist_ok=True)
        info_list = []
        info_list_2 = []

        for file in os.listdir(img_dir):
            path = os.path.join(img_dir, file)
            img = cv2.imread(path)
            if img is None:
                continue

            results = self.model(img, size=640)
            results.render()
            out_img = results.ims[0]

            detections = results.pandas().xyxy[0]
            target_labels = self.target_map.get(self.target_type, ["person"])
            count = detections[detections['name'].isin(target_labels)].shape[0]

            if count >= self.threshold:
                out_path = os.path.join(output_dir, file)
                cv2.imwrite(out_path, out_img)
                info_list.append({
                    "file_name": file,
                    "num_persons": int(count)
                })

            info_list_2.append({
                "file_name": file,
                "num_persons": int(count)
            })

        with open(info_file, "w") as f:
            json.dump(info_list, f, indent=2)

        return info_list, info_list_2
