# app/main.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.train_model import SimpleNN
import io
import os
import gdown

app = FastAPI()

MODEL_PATH = "model/model.pth"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1GiTEqFEY_pfVvasgBEslJ6vB_gSDeQWv"

model = None


# Load model
@app.on_event("startup")
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already present.")

    global model
    model = SimpleNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()


# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),  # In case input is RGB
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
