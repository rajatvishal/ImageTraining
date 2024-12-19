from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import json
import torchvision.transforms as T
from PIL import Image
import torch
from pycocotools.coco import COCO

class CustomDataset(Dataset):
    def __init__(self, images_path, annotations_path, transforms=None):
        self.coco = COCO(annotations_path)
        self.images_path = images_path
        self.transforms = transforms

    def __getitem__(self, idx):
        # Load image and annotation info from COCO
        image_info = self.coco.loadImgs(ids=[idx])[0]
        image_path = os.path.join(self.images_path, image_info["file_name"])

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # Load annotations for the image
        annotation_ids = self.coco.getAnnIds(imgIds=[image_info["id"]])
        annotations = self.coco.loadAnns(annotation_ids)

        # Prepare the targets for Mask R-CNN
        target = {
            "boxes": torch.tensor([ann["bbox"] for ann in annotations], dtype=torch.float32),
            "labels": torch.tensor([ann["category_id"] for ann in annotations], dtype=torch.int64),
            "masks": torch.tensor([ann["segmentation"] for ann in annotations], dtype=torch.uint8),
            "image_id": torch.tensor([image_info["id"]]),
            "area": torch.tensor([ann["area"] for ann in annotations], dtype=torch.float32),
            "iscrowd": torch.tensor([ann["iscrowd"] for ann in annotations], dtype=torch.uint8)
        }

        return image, target

    def __len__(self):
        return len(self.coco.getImgIds())

def get_dataloader(config):
    datasets = []
    for dataset_name, dataset_info in config["datasets"].items():
        dataset = CustomDataset(
            images_path=os.path.join(dataset_info["path"], "images"),
            annotations_path=os.path.join(dataset_info["path"], "datasets_a/annotations.json"),
            transforms=T.Compose([T.ToTensor()])
        )
        datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    return DataLoader(combined_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
