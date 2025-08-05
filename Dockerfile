# Use official Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (better cache usage)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else into the container
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
