import torch
from models.mask_rcnn import get_model_instance_segmentation
from src.data_loader import get_dataloader
from torch.optim import Adam
import os

# Set device to CPU
device = torch.device("cpu")

def train_model(config):
    dataloader = get_dataloader(config)

    # Initialize the model
    model = get_model_instance_segmentation(config["dataset"]["num_classes"])

    # Move model to CPU
    model.to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config["train"]["learning_rate"])

    model.train()
    for epoch in range(config["train"]["num_epochs"]):
        for images, targets in dataloader:
            # Move images and targets to CPU
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Total loss
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['train']['num_epochs']}, Loss: {losses.item()}")

        # Save model checkpoint after every epoch
        checkpoint_path = os.path.join(config["train"]["checkpoint_dir"], f"checkpoint_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    config = {}  # Load your config here
    train_model(config)
