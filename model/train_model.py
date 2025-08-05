# Full example coming up step by step. We start with training a simple classifier using PyTorch on MNIST.

# STEP 1: TRAIN A SIMPLE MODEL

# File: train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

def train():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = SimpleNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(10):  # Train for 10 epochs
        print(f"Epoch {epoch + 1}")
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")

    torch.save(model.state_dict(), "model.pth")
    print("Model trained and saved.")

if __name__ == "__main__":
    train()
