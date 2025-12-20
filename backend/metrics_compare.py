import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    roc_auc_score
)
from PIL import Image
import cv2

# --- FRAMEWORK IMPORTS ---
import torch
import torch.nn as nn
from torchvision import models, transforms
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50

# ----------------------------------------
# 1. SHARED CONFIGURATION & DATA SCANNERS
# ----------------------------------------
def get_image_paths(data_dir, class_names):
    """
    Scans directory to get a consistent list of (path, label).
    It looks for folders matching the provided class_names.
    """
    paths = []
    labels = []
    
    print(f"Scanning: {data_dir}")
    print(f"Looking for folders containing: {class_names}")
    
    # Debug counter to see what we find
    folders_seen = set()
    
    for root, dirs, files in os.walk(data_dir):
        parent_folder = os.path.basename(root)
        folders_seen.add(parent_folder)
        
        # Determine Label based on arguments
        label = -1
        
        # Check if parent folder matches our class names (case-insensitive)
        for i, class_name in enumerate(class_names):
            if class_name.lower() in parent_folder.lower():
                label = i
                break
        
        # Special handling for "TB" abbreviation if user passed "Tuberculosis"
        if label == -1 and "Tuberculosis" in class_names[1] and "TB" in parent_folder:
             label = 1

        if label != -1:
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(root, file)
                    paths.append(full_path)
                    labels.append(label)
                    
    if len(paths) == 0:
        print("\nNO IMAGES FOUND!")
        print(f"   I searched inside: {data_dir}")
        print(f"   I saw these folders: {list(folders_seen)[:5]} ...")
        print("   Make sure your --data-dir points to the folder containing 'Normal' and 'Tuberculosis' subfolders.")
        return np.array([]), np.array([])
        
    print(f"Found {len(paths)} images.")
    return np.array(paths), np.array(labels)

# ----------------------------------------
# 2. MODEL A: PYTORCH SPECIFICS
# ----------------------------------------
class PyTorchHandler:
    def __init__(self, weights_path, device, num_classes):
        self.device = device
        self.weights = weights_path
        self.num_classes = num_classes
        self.model = self._build_model()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        print("[Model A] Building PyTorch ResNet50...")
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        try:
            state = torch.load(self.weights, map_location=self.device)
            model.load_state_dict(state)
            print(f"[Model A] Weights loaded: {self.weights}")
        except Exception as e:
            print(f"[Model A] Load failed: {e}")
        
        model.to(self.device)
        model.eval()
        return model

    def predict(self, paths):
        print("[Model A] Starting Inference...")
        probs = []
        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            for i, path in enumerate(paths):
                if i % 100 == 0: print(f"   A: Processing {i}/{len(paths)}", end='\r')
                try:
                    img = Image.open(path).convert("RGB")
                    img_t = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # TTA: Original + Flip
                    out1 = self.model(img_t)
                    prob1 = softmax(out1)
                    
                    img_flip = torch.flip(img_t, [3])
                    out2 = self.model(img_flip)
                    prob2 = softmax(out2)
                    
                    avg = (prob1 + prob2) / 2
                    probs.append(avg[0, 1].item())
                except Exception as e:
                    print(f"Err {path}: {e}")
                    probs.append(0.0)
        return np.array(probs)

# ----------------------------------------
# 3. MODEL B: KERAS SPECIFICS
# ----------------------------------------
class KerasHandler:
    def __init__(self, weights_path):
        self.weights = weights_path
        self.model = self._build_model()
        self.img_size = 256

    def _build_model(self):
        print("\n[Model B] Building Keras ResNet50...")
        base_model = ResNet50(weights=None, include_top=False, input_shape=(256, 256, 3))
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        try:
            model.load_weights(self.weights)
            print(f"[Model B] Weights loaded: {self.weights}")
        except Exception as e:
            try:
                model = load_model(self.weights, compile=False)
                print(f"[Model B] Loaded via load_model.")
            except:
                print(f"[Model B] Load failed: {e}")
        return model

    def predict(self, paths):
        print("[Model B] Starting Inference...")
        probs = []
        
        for i, path in enumerate(paths):
            if i % 100 == 0: print(f"   B: Processing {i}/{len(paths)}", end='\r')
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: raise Exception("Img not found")
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)

                p1 = self.model.predict(img, verbose=0)[0][0]
                img_flip = np.flip(img, axis=2)
                p2 = self.model.predict(img_flip, verbose=0)[0][0]
                
                probs.append((p1 + p2) / 2)
            except Exception as e:
                print(f"Err {path}: {e}")
                probs.append(0.0)
        return np.array(probs)

# ----------------------------------------
# 4. OPTIMIZER & EVALUATOR
# ----------------------------------------
def find_best_threshold(y_true, y_scores, model_name, search_space):
    print(f"\nOptimization for {model_name}")
    print(f"{'THRESH':<8} | {'NORM PREC':<10} | {'TB RECALL':<10} | {'F1 SCORE':<10}")
    print("-" * 50)
    
    best_f1 = 0
    best_thresh = 0.5
    
    for t in search_space:
        preds = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
        
        norm_prec = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        tb_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (tb_prec * recall) / (tb_prec + recall) if (tb_prec + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
        if t in [0.01, 0.05, 0.10, 0.50, 0.80]:
             print(f"{t:.2f}     | {norm_prec:.4f}     | {recall:.4f}     | {f1:.4f}")
             
    print(f"Best F1 Thresh for {model_name}: {best_thresh:.4f} (F1: {best_f1:.4f})")
    return best_thresh

def plot_comparison(y_true, scores_a, scores_b, thresh_a, thresh_b, class_names):
    # 1. ROC Curve
    try:
        plt.figure(figsize=(10, 6))
        fpr_a, tpr_a, _ = roc_curve(y_true, scores_a)
        auc_a = roc_auc_score(y_true, scores_a)
        
        fpr_b, tpr_b, _ = roc_curve(y_true, scores_b)
        auc_b = roc_auc_score(y_true, scores_b)
        
        plt.plot(fpr_a, tpr_a, label=f'Model A (PyTorch) AUC={auc_a:.3f}', color='blue')
        plt.plot(fpr_b, tpr_b, label=f'Model B (Keras) AUC={auc_b:.3f}', color='red', linestyle='--')
        plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
        plt.title("ROC Curve Comparison")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.savefig("comparison_roc.png")
        print("Saved 'comparison_roc.png'")
        
        # 2. Confusion Matrices Side-by-Side
        preds_a = (scores_a >= thresh_a).astype(int)
        preds_b = (scores_b >= thresh_b).astype(int)
        
        cm_a = confusion_matrix(y_true, preds_a)
        cm_b = confusion_matrix(y_true, preds_b)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                    xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title(f"Model A (Thresh {thresh_a:.2f})")
        axes[0].set_ylabel("Actual")
        axes[0].set_xlabel("Predicted")

        sns.heatmap(cm_b, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title(f"Model B (Thresh {thresh_b:.2f})")
        axes[1].set_xlabel("Predicted")
        
        plt.savefig("comparison_cm.png")
        print("Saved 'comparison_cm.png'")
    except Exception as e:
        print(f"Could not save plots: {e}")

# ----------------------------------------
# 5. MAIN EXECUTION
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--weights-pt", type=str, required=True, help="Path to PyTorch .pth")
    parser.add_argument("--weights-tf", type=str, required=True, help="Path to Keras .keras/.h5")
    parser.add_argument("--class-names", type=str, nargs="+", default=["Normal", "Tuberculosis"])
    args = parser.parse_args()

    # 1. Get Data
    paths, y_true = get_image_paths(args.data_dir, args.class_names)
    if len(paths) == 0: return

    # 2. Run Model A (PyTorch)
    print("\n--- STAGE 1: PyTorch Evaluation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    handler_a = PyTorchHandler(args.weights_pt, device, len(args.class_names))
    scores_a = handler_a.predict(paths)
    del handler_a
    torch.cuda.empty_cache()
    
    # 3. Run Model B (Keras)
    print("\n--- STAGE 2: Keras Evaluation ---")
    handler_b = KerasHandler(args.weights_tf)
    scores_b = handler_b.predict(paths)
    del handler_b
    
    # 4. Optimization
    # Model A: Fine search (0.01 - 0.55)
    space_a = np.concatenate([np.arange(0.01, 0.10, 0.01), np.arange(0.10, 0.55, 0.05)])
    thresh_a = find_best_threshold(y_true, scores_a, "Model A", space_a)
    
    # Model B: Full spectrum (0.01 - 0.99)
    space_b = np.arange(0.01, 0.99, 0.05)
    thresh_b = find_best_threshold(y_true, scores_b, "Model B", space_b)
    
    # 5. Final Reporting
    print(f"\n{'='*30} COMPARISON SUMMARY {'='*30}")
    
    for name, scores, thresh in [("Model A (PyTorch)", scores_a, thresh_a), 
                                 ("Model B (Keras)", scores_b, thresh_b)]:
        preds = (scores >= thresh).astype(int)
        
        report = classification_report(y_true, preds, target_names=args.class_names, output_dict=True)
        # Assuming args.class_names[0] is the "Normal" class
        norm_prec = report[args.class_names[0]]['precision']
        
        print(f"\n🔹 {name}")
        print(f"   Selected Threshold: {thresh:.4f}")
        print(f"   Accuracy:           {report['accuracy']:.4f}")
        print(f"   {args.class_names[0]} Precision:   {norm_prec:.4f}")
        print(f"   {args.class_names[1]} Recall:      {report[args.class_names[1]]['recall']:.4f}")
        print(f"   F1 Score ({args.class_names[1]}):      {report[args.class_names[1]]['f1-score']:.4f}")
        print(f"   ROC AUC:            {roc_auc_score(y_true, scores):.4f}")

    # 6. Plotting
    plot_comparison(y_true, scores_a, scores_b, thresh_a, thresh_b, args.class_names)

    # 7. Save CSV
    df = pd.DataFrame({
        "path": paths,
        "true_label": [args.class_names[i] for i in y_true],
        "prob_model_A": scores_a,
        "prob_model_B": scores_b,
        "pred_A": [args.class_names[i] for i in (scores_a >= thresh_a).astype(int)],
        "pred_B": [args.class_names[i] for i in (scores_b >= thresh_b).astype(int)]
    })
    df.to_csv("comparison_results.csv", index=False)
    print("\nDetailed results saved to 'comparison_results.csv'")

if __name__ == "__main__":
    main()