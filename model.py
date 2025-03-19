import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import UAVDataset
from utils import DiceLoss, dice_score
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE
import logging

logging.basicConfig(level=logging.INFO)

def get_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 1, 1)
    )

def train_model():
    dataset = UAVDataset(["seq14_000200_0_5.png"])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model().to(DEVICE)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.permute(0, 3, 1, 2).to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    logging.info("Model saved as model.pth")

def evaluate_model():
    dataset = UAVDataset(["seq14_000200_0_5.png"])
    dataloader = DataLoader(dataset, batch_size=1)
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.permute(0, 3, 1, 2).to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)
            outputs = model(images)
            score = dice_score(outputs, masks)
            logging.info(f"Dice Score: {score:.4f}")