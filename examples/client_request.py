import requests

# Path to test image
image_path = "10006.png"  # Example MNIST digit image (1)

# Endpoint URL
url = "http://127.0.0.1:8000/predict"

# Send POST request
with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Print prediction
print("Response:", response.json())
