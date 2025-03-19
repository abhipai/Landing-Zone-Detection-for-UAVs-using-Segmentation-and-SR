import torch
import numpy as np
import cv2
from utils import DiceLoss, dice_score, apply_morphological_opening, compute_distance_transform
from dataset import UAVDataset
from config import IMAGE_DIR

def test_dice_loss():
    pred = torch.rand((1, 1, 256, 256))
    target = torch.rand((1, 1, 256, 256))
    criterion = DiceLoss()
    loss = criterion(pred, target)
    assert loss.item() >= 0, "Dice Loss should be non-negative"

def test_dice_score():
    pred = torch.rand((1, 1, 256, 256))
    target = torch.rand((1, 1, 256, 256))
    score = dice_score(pred, target)
    assert 0 <= score <= 1, "Dice Score should be between 0 and 1"

def test_dataset_loading():
    dataset = UAVDataset(["seq14_000200_0_5.png"])
    image, mask = dataset[0]
    assert image.shape == (256, 256, 3) or image.shape[-1] == 3, "Image should have 3 channels"
    assert mask.ndim == 2 or mask.ndim == 3, "Mask should be 2D or 3D array"

def test_postprocessing_steps():
    dummy_img = np.zeros((256, 256), dtype=np.uint8)
    dummy_img[100:150, 100:150] = 255
    morph_img = apply_morphological_opening(dummy_img)
    dist_img = compute_distance_transform(morph_img)
    assert dist_img.shape == dummy_img.shape, "Distance transform output should match input shape"

if __name__ == "__main__":
    test_dice_loss()
    test_dice_score()
    test_dataset_loading()
    test_postprocessing_steps()
    print("All tests passed!")