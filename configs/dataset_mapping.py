from detectron2.data.datasets import register_coco_instances
import os

def register_datasets():
    datasets = [
        ("datasets_a", "./datasets/datasets_a/annotations/annotations.json", "./datasets/datasets_a/images"),
        ("datasets_b", "./datasets/datasets_b/annotations/annotations.json", "./datasets/datasets_b/images"),
        ("combined_validation", "./datasets/combined_validation/annotations/annotations.json", "./datasets/combined_validation/images")
    ]

    for dataset_name, annotations_path, images_path in datasets:
        if not os.path.exists(annotations_path):
            print(f"Error: Annotations for {dataset_name} not found at {annotations_path}")
            continue
        if not os.path.exists(images_path):
            print(f"Error: Image directory for {dataset_name} not found at {images_path}")
            continue
        # Register each dataset
        register_coco_instances(dataset_name, {}, annotations_path, images_path)
        print(f"Registered {dataset_name} successfully.")

if __name__ == "__main__":
    register_datasets()