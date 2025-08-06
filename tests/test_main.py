from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io


client = TestClient(app)


def test_predict_endpoint():
    # Create a dummy 28x28 grayscale image
    image = Image.new("L", (28, 28), color=255)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Send as multipart/form-data
    response = client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)
