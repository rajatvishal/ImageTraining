import torch
import matplotlib.pyplot as plt
from src.data_loader import get_dataloader
from models.mask_rcnn import get_model_instance_segmentation

# Set device to CPU
device = torch.device("cpu")

def test_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            predictions = model(images)

            for i in range(len(predictions)):
                plt.imshow(images[i].cpu().permute(1, 2, 0).numpy())
                plt.show()

                # Visualize masks, boxes, and labels
                print(predictions[i])

if __name__ == "__main__":
    # Load the pre-trained model and checkpoint
    config = {
        "dataset": {
            "test_data_dir": "path/to/test_data",  # Path to the test dataset (adjust as needed)
            "num_classes": 4,  # Number of classes (including background)
        },
        "test": {
            "batch_size": 2,  # Batch size for inference
        }
    }

    model = get_model_instance_segmentation(4)  # Assuming 4 classes (including background)
    model.load_state_dict(torch.load("path/to/checkpoint.pth", map_location=device))

    # Set model to CPU
    model.to(device)

    # Get the test dataloader
    test_dataloader = get_dataloader(config)


    test_model(model, get_dataloader(config))
