import torchvision
import torch
import torch.nn as nn
import torchvision.models.detection.mask_rcnn

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained Mask R-CNN model from torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Modify the classifier to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Modify the mask predictor to match the number of classes
    in_mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_mask_features, 256, num_classes)
    
    return model
