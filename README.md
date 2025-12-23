# Early TB Detector using ResNet50

A full-stack medical AI application designed to detect Tuberculosis (TB) from Chest X-ray images. This project utilizes a **ResNet50** convolutional neural network (PyTorch) for classification, served via a **FastAPI** backend, and consumed by a modern **React** frontend.

## 🔗 Project Notebook
You can view the training process, data analysis, and model prototyping in our Google Colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AlXLHPxSGMzFeX8vgbcmnf_0iOnZhus0?usp=sharing)

## 📂 Project Structure

- **`backend/`**: Contains the Python/FastAPI server, model evaluation scripts (`metrics.py`, `metrics_compare.py`), and PyTorch model logic.
- **`frontend/`**: Contains the React web application with a modern UI for user interaction.
- **`Dataset.../`**: (Local) Directory containing the chest radiography image database.

## 🚀 Quick Start

To run the full application, you need to start both the backend server and the frontend client.

### Prerequisites
- Python 3.9+
- Node.js & npm
- CUDA (Optional, for GPU acceleration)

### 1. Start the Backend
Navigate to the backend folder and start the FastAPI server:

```bash
cd backend
# Install dependencies (see backend/README.md)
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 2. Start the Frontend
Open a new terminal, navigate to the frontend folder, and start the React app:

```bash
cd frontend
npm install
npm run dev
```

The web interface will typically launch at `http://localhost:5173`.

## 🧠 Model Architecture
The core model is a fine-tuned **ResNet50** architecture.
- **Input:** 224x224 RGB Images.
- **Backbone:** ResNet50 (Pre-trained on ImageNet).
- **Head:** Custom fully connected layers with Dropout (0.4-0.5) and ReLU.
- **Inference Strategy:** Uses Test Time Augmentation (TTA) by averaging predictions of the original image and a horizontally flipped version.
