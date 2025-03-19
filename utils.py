import numpy as np
import cv2
import torch
import torch.nn as nn

def apply_morphological_opening(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def compute_distance_transform(binary_image):
    return cv2.distanceTransform(binary_image, cv2.DIST_L2, 5, cv2.DIST_MASK_PRECISE)

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def dice_score(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    return dice.item()