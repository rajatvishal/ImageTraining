import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.mask_rcnn import MaskRCNNModel
from configs.dataset_mapping import register_datasets

def main():
    register_datasets()

    model = MaskRCNNModel("./configs/mask_rcnn_config.yaml")

    # Load trained weights
    model.cfg.MODEL.WEIGHTS = "./result/checkpoints/"  # Adjust path to your trained model

    results = model.test("combined_validation")
    print(results)

if __name__ == "__main__":
    main()