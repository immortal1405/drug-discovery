# Molecular Generation AI Platform

An automated multi-objective molecular generation platform with explainability for drug discovery.

## Features

- Multi-objective molecular generation using VAE, GAN, and GNN models
- Property prediction (binding affinity, solubility, toxicity)
- Explainable AI (XAI) for molecular rationalization
- Cloud-based high-throughput virtual screening
- Reinforcement learning for model refinement

## Prerequisites

- Python 3.9+
- Google Cloud Platform account
- Docker
- Kubernetes (for local development)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd molecular-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Project Structure

```
molecular-ai/
├── docker/              # Docker configuration
├── src/                 # Source code
│   ├── models/         # Model implementations
│   ├── training/       # Training scripts
│   ├── evaluation/     # Evaluation scripts
│   └── utils/          # Utility functions
├── tests/              # Test files
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Usage

### Training

To train a model:

```bash
python src/train.py
```

### Evaluation

To evaluate a model:

```bash
python src/evaluate.py
```

### API Server

To start the API server:

```bash
python src/api.py
```

## Deployment

### Building Docker Image

```bash
docker build -t molecular-ai -f docker/Dockerfile .
```

### Deploying to GCP

1. Push the Docker image to Container Registry:
```bash
gcloud builds submit --tag gcr.io/moleculargeneration/molecular-ai
```

2. Deploy to Vertex AI:
```bash
gcloud ai models upload --model=molecular_generation --container-image-uri=gcr.io/moleculargeneration/molecular-ai
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 