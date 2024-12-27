import sys
import os
import torch  # Required to check device availability

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.mask_rcnn import MaskRCNNModel
from configs.dataset_mapping import register_datasets
from datasets.preprocess import filter_annotations, resize_images
 
def main():

    try:
        register_datasets()
    except Exception as e:
        print(f"Error while registering datasets: {e}")
        return

    category_filter = ["person", "car", "bicycle", "dog", "cat"]  # Add more categories here
    
    try: 
        resize_images("./datasets/datasets_a/images", "./datasets/datasets_a/resized_images")
    except Exception as e:
        print(f"Error while Resize Image: {e}")
        return

    try:
        filter_annotations(
            "./datasets/datasets_a/annotations/annotations.json",
            "./datasets/datasets_a/annotations/filtered_annotations.json",
            category_filter
        )
    except Exception as e:
        print(f"Error while Filter Annotations: {e}")
        return

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    OUTPUT_DIR = "./result"
    CONFIG_PATH = "./configs/mask_rcnn_config.yaml"

    # Initialize the model
    try:
        model = MaskRCNNModel(config_path=CONFIG_PATH)
        model.train(device=device, output_dir=OUTPUT_DIR)
    except Exception as e:
        print(f"Error while Initialize the model {e}")  

if __name__ == "__main__":
    main()