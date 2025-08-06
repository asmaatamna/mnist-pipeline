import torch
from model.train_model import SimpleNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def test_model_forward_shape():
    model = SimpleNN()
    sample_input = torch.randn(1, 1, 28, 28)  # single MNIST-like image
    output = model(sample_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"

def test_training_on_small_batch():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    small_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for images, labels in small_loader:
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        break  # only 1 batch for the test

    assert loss.item() > 0, "Loss should be a positive number"
