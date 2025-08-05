# app/main.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.train_model import SimpleNN
import io
import os

app = FastAPI()

model = None  # Global variable for lazy loading

@app.on_event("startup")
def load_model():
    global model
    model = SimpleNN()
    model_path = "model/model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    else:
        print("Warning: model.pth not found. /predict will not work.")

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),     # In case input is RGB
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded."}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return {"prediction": predicted.item()}
