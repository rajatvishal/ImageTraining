import os
import json
from PIL import Image  # To get the image dimensions

def convert_to_coco(dataset_path, output_path, categories):
    """
    Convert a custom dataset to COCO format.

    Args:
    - dataset_path: The root folder of the dataset.
    - output_path: Path where the converted annotations file will be saved.
    - categories: List of category names.
    """
    images = []
    annotations = []
    category_mapping = {name: idx + 1 for idx, name in enumerate(categories)}

    image_id = 0
    annotation_id = 0

    images_path = os.path.join(os.getcwd(), dataset_path, "images")  
    
    for file_name in os.listdir(images_path):
        if file_name.endswith((".jpg", ".png")):
            image_id += 1
            image_path = os.path.join(images_path, file_name)
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })

            # Example: if you have multiple annotations per image, you need to loop and add them.
            # For now, it's just one annotation as an example.
            # Replace this logic to iterate over your actual annotations.

            # Example of how you might parse your own bounding boxes
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "bbox": [50, 50, 200, 200],  # Example bounding box, replace with real data
                "area": 200 * 200,  # Replace with real area
                "category_id": category_mapping["Iron"],  # Replace with correct category
                "iscrowd": 0
            })
            annotation_id += 1

    # Create categories for the COCO format
    coco_categories = [{"id": idx, "name": name} for name, idx in category_mapping.items()]

    # Create the final COCO formatted dictionary
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories
    }

    # Write the output JSON file
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=4)

# Example usage:
convert_to_coco("datasets/datasets_a/", "datasets/datasets_a/annotations/annotations.json", ["background", "Iron", "Steel", "Copper", "Aluminum", "Oil Cans", "Plastic Bottle"])
convert_to_coco("datasets/datasets_b/", "datasets/datasets_b/annotations/annotations.json", ["background", "Iron", "Steel", "Copper", "Aluminum", "Oil Cans", "Plastic Bottle"])
