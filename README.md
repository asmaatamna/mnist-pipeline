# MNIST Classifier Deployment

This project demonstrates how to train a simple image classifier on the MNIST dataset and deploy it as a web service using FastAPI and Docker. It also includes unit and integration tests, a basic CI/CD pipeline with GitHub Actions, and optional containerization for production use.

## Project Structure

- `model/`: Training script and saved model
- `app/`: FastAPI application for serving predictions
- `tests/`: Unit and integration tests
- `.github/workflows/`: CI/CD pipeline (GitHub Actions)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python model/train_model.py

# Start the FastAPI server
uvicorn app.api:app --reload
