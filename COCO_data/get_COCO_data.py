from pycocotools.coco import COCO
import requests
import os
import json

# COCO 2017 annotationed data record, we use as parameters to downlaod data using COCO api
json_file = open("./get_coco_data.json", "r")
parameter_dict = json.load(json_file)

annFile = parameter_dict["annotations_file"]
coco = COCO(annFile)

# get catagroy IDs
data_type = parameter_dict["data_type"]

if data_type == "pets":
    label_list = ['cat', 'dog']   # pets
elif data_type == "people":
    label_list = ['person']    # people
elif data_type == "vehicles":
    label_list = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']  # vehicles
elif data_type == "train":
    label_list = ['train'] # railroad trains 
elif data_type == "airplane":
    label_list = ['airplane']  # airplane
else:
    print("Sorry we only want the data related to people, pets, vehicles, and train. Maybe more labels in future version")

person_cat_id = coco.getCatIds(catNms=label_list)[0]

# get data of corrspond label
img_ids_with_person = coco.getImgIds(catIds=[person_cat_id])

total_pictures = parameter_dict["total_pictures"]
obj_in_picture = parameter_dict["obj_in_picture"]

# find the picture contains at least 2 objects
multi_person_img_ids = []
for img_id in img_ids_with_person:
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[person_cat_id])
    if len(ann_ids) >= obj_in_picture:
        multi_person_img_ids.append(img_id)
    if len(multi_person_img_ids) >= total_pictures:
        break

# load inages
images = coco.loadImgs(multi_person_img_ids)

# save to a folder for testing of Yolo img detection platform
save_dir = parameter_dict["savedir_frontPart"] + data_type
os.makedirs(save_dir, exist_ok=True)

print("downloading...")

for img in images:
    img_url = img['coco_url']
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(save_dir, img['file_name']), 'wb') as f:
            f.write(img_data)
        print(f"Downloaded {img['file_name']}")
    except Exception as e:
        print(f"Failed to download {img['file_name']}: {e}")
