import torch
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from datetime import datetime


# Loading the YOLOv5 model using torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Setting path of test_images directory
images_path = Path(
    '/Object Detection using YoloV5 pretrained model/tes_images/images')

# Get a list of image files in the directory
image_files = list(images_path.glob('*.jpg'))


detection_results = {
    "images": [],
    "categories": [{"id": 0, "name": "Person"}],
    "annotations": [],
    "info": {
        "year": datetime.now().year,
        "version": "1.0",
        "description": "",
        "contributor": "Rahul PS",
        "url": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    }
}


# Perform object detection on each image
for image_id, image_file in enumerate(tqdm(image_files, desc='Detecting objects')):
    img = Image.open(image_file)
    results = model(img)  # doing inference with each image

    # Get bounding box coordinates (xyxy) -- pretrained cordinates of YOLOv5
    # cpu -- identifies the dynamic compatability of model in torch
    bboxes = results.xyxy[0].cpu().numpy().tolist()
    num_objects = len(bboxes)

    detection_results["images"].append({
        "width": img.width,
        "height": img.height,
        "frame_number": image_id,
        "num_objects": num_objects,
        "image_name": str(image_file)

    })

    for idx, (bbox, label) in enumerate(zip(bboxes, results.xyxy[0][:, -1].cpu().numpy())):
        detection_results["annotations"].append({
            "id": len(detection_results["annotations"]),
            "image_id": image_id,
            "category_id": int(label),
            "segmentation": [],
            "bbox": bbox,
            "ignore": 0,
            "iscrowd": 0,
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        })


# Saving detection results to a JSON file
output_json_path = 'detection_results.json'
with open(output_json_path, 'w') as json_file:
    json.dump(detection_results, json_file, indent=2)

print(f'Detection results saved to {output_json_path}')
