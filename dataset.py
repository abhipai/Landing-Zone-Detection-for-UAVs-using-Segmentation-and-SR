import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import IMAGE_ROOT_DIR, MASK_ROOT_DIR, SRGAN_WEIGHTS
import logging
from super_resolution import load_srgan_model, enhance_image

logging.basicConfig(level=logging.INFO)

srgan_model = load_srgan_model(SRGAN_WEIGHTS)

def collect_image_mask_pairs():
    image_paths = []
    mask_paths = []
    for seq in os.listdir(IMAGE_ROOT_DIR):
        seq_image_dir = os.path.join(IMAGE_ROOT_DIR, seq, "Images")
        seq_mask_dir = os.path.join(MASK_ROOT_DIR, seq, "Masks")
        if not os.path.isdir(seq_image_dir):
            continue
        for img_name in os.listdir(seq_image_dir):
            image_paths.append(os.path.join(seq_image_dir, img_name))
            mask_paths.append(os.path.join(seq_mask_dir, img_name))
    return image_paths, mask_paths

class UAVDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = enhance_image(srgan_model, image)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image / 255.0
        mask = mask / 255.0
        return image.astype(np.float32), mask.astype(np.float32)