# Backend - TB Detection API & Metrics

This directory contains the machine learning logic, evaluation scripts, and the FastAPI server that powers the Early TB Detector.

## 🛠️ Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   **Option A: Manual Installation (Flexible/Lightweight)**
   Use this for custom control or to install the CPU-only version of PyTorch to save space:
   ```bash
   # For CPU-only (Recommended for standard laptops)
   pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
   
   # Then install remaining dependencies
   pip install fastapi uvicorn numpy pandas matplotlib seaborn scikit-learn python-multipart opencv-python tensorflow
   ```

   **Option B: Standard Installation (Using requirements.txt)**
   Best for a quick, complete setup:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: TensorFlow is only required if running `metrics_compare.py`)*

3. **Model Weights:**
   Ensure your trained model file (`tb_resnet50_robust.pth`) is placed in this `backend/` directory.

## 📡 API Usage (`main.py`)

The API is built with FastAPI. It loads the PyTorch model and provides an inference endpoint for the frontend.

**Run the server:**
```bash
uvicorn main:app --reload
```

**Endpoints:**
- `POST /predict`: Upload an image file to receive a TB prediction, confidence score, and boolean flag.

## 📊 Evaluation & Metrics Scripts

### 1. Single Model Evaluation (`metrics.py`)
Evaluates the PyTorch model against a dataset or CSV using **Test Time Augmentation (TTA)** and performs **Threshold Tuning** to find the optimal F1 score.



```bash
python metrics.py --weights tb_resnet50_robust.pth --data-dir path/to/dataset --save-csv predictions.csv
```
*Outputs:*
- `threshold_analysis.png`: A chart showing the trade-off between Precision, Recall, and F1 at different thresholds.
- `predictions_tuned.csv`: Detailed predictions for every image.

### 2. Framework Comparison (`metrics_compare.py`)
This script compares the performance of the **PyTorch** model against a **Keras/TensorFlow** version of the same architecture side-by-side.

```bash
python metrics_compare.py \
  --data-dir path/to/dataset \
  --weights-pt tb_resnet50_robust.pth \
  --weights-tf tb_resnet50_keras.h5
```
*Outputs:*
- `comparison_roc.png`: ROC Curve overlay for both models.
- `comparison_cm.png`: Side-by-side Confusion Matrices.
- `comparison_results.csv`: Raw probability outputs for both models.

## ⚙️ Key Features
- **TTA (Test Time Augmentation):** Inferences are averaged between the original image and a horizontally flipped version to improve robustness.
- **Threshold Optimization:** The system automatically calculates the optimal probability threshold (often between 0.01 and 0.50) to balance Safety (Normal Precision) and Detection (TB Recall).