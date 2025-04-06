FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install RDKit
RUN pip install --no-cache-dir rdkit

# Install PyTorch and related packages
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install DeepChem
RUN pip install --no-cache-dir deepchem

# Install AutoDock Vina
RUN apt-get update && apt-get install -y \
    autodock-vina \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Command to run the training
CMD ["python", "src/train.py"] 