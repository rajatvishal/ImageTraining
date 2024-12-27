import os
import json
import cv2
from tqdm import tqdm

def resize_images(image_dir, output_dir, size=(800, 800)):
    os.makedirs(output_dir, exist_ok=True)
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        if img is not None:
            resized_img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(output_dir, image_name), resized_img)

def filter_annotations(annotation_path, output_path, category_filter):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    filtered_annotations = {
        "images": annotations["images"],
        "categories": [cat for cat in annotations["categories"] if cat["name"] in category_filter],
        "annotations": [
            ann for ann in annotations["annotations"] if ann["category_id"] in [
                cat["id"] for cat in annotations["categories"] if cat["name"] in category_filter
            ]
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(filtered_annotations, f, indent=4)

if __name__ == "__main__":
    category_filter = ["person", "car", "bicycle", "dog", "cat"]  # Add more categories here
    resize_images("./datasets/datasets_a/images", "./datasets/datasets_a/resized_images")
    filter_annotations(
        "./datasets/datasets_a/annotations/annotations.json",
        "./datasets/datasets_a/annotations/filtered_annotations.json",
        category_filter
    )