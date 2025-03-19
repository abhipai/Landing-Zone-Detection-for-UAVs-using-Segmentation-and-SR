import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2

class FastSRGAN(nn.Module):
    def __init__(self):
        super(FastSRGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, padding=4)
        )

    def forward(self, x):
        return self.model(x)

def load_srgan_model(weights_path):
    model = FastSRGAN()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def enhance_image(model, image_np):
    transform = transforms.ToTensor()
    image_tensor = transform(image_np).unsqueeze(0)
    with torch.no_grad():
        sr_image = model(image_tensor)
    sr_image = sr_image.squeeze(0).permute(1, 2, 0).numpy()
    sr_image = (sr_image * 255.0).clip(0, 255).astype('uint8')
    return sr_image