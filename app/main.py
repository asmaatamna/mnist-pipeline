# app/main.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.train_model import SimpleNN
import io

app = FastAPI()

# Load model
model = SimpleNN()
model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),     # In case input is RGB
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return {"prediction": predicted.item()}
