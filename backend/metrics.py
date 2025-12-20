import torch
import os
import argparse
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ----------------------------------------
# Configuration
# ----------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------
# Helper Dataset
# ----------------------------------------
class ImageCSV(Dataset):
    def __init__(self, csv_path, root_dir="", transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row.iloc[0] 
        label = row.iloc[1]
        path = os.path.join(self.root, fname) if self.root else fname
        
        try:
            image = Image.open(path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, label, str(path)

# ----------------------------------------
# Model Loader
# ----------------------------------------
def build_model(num_classes=2, device="cpu", weights_load_path=None):
    model = models.resnet50(weights=None)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    if weights_load_path:
        state = torch.load(weights_load_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights: {weights_load_path}")
        
    model.to(device)
    model.eval()
    return model

# ----------------------------------------
# Evaluation (Threshold Tuner)
# ----------------------------------------
def evaluate(model, dataloader, device, class_names, save_csv_path=None):
    print(f"\n--- Evaluation Started (TTA Enabled) ---")
    
    all_true = []
    all_probs = []
    all_paths = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0: print(f"Processing batch {i}...", end='\r')
            
            if len(batch) == 3: imgs, labels, paths = batch
            else: imgs, labels = batch; paths = [""] * len(labels)

            imgs = imgs.to(device)

            # --- TTA: FLIP & AVERAGE ---
            out1 = model(imgs)
            prob1 = softmax(out1)

            imgs_flip = torch.flip(imgs, [3])
            out2 = model(imgs_flip)
            prob2 = softmax(out2)

            avg_prob = (prob1 + prob2) / 2
            
            tb_probs = avg_prob[:, 1].cpu().numpy()
            
            for k in range(len(labels)):
                lab = labels[k].item() if isinstance(labels[k], torch.Tensor) else labels[k]
                all_true.append(lab)
                all_probs.append(float(tb_probs[k]))
                all_paths.append(paths[k])

    y_true = np.array(all_true)
    y_scores = np.array(all_probs)

    # ------------------------------------------------------
    # THE THRESHOLD OPTIMIZER (Grid Search)
    # ------------------------------------------------------
    print(f"\n\n{'='*75}")
    print(f"{'THRESH':<8} | {'NORM PREC':<10} | {'TB RECALL':<10} | {'F1 SCORE':<10} | {'TB PREC':<10}")
    print(f"{'='*75}")
    
    # We search specifically in the "low threshold" zone where the action happens
    # (0.01 to 0.50)
    search_space = np.concatenate([
        np.arange(0.01, 0.10, 0.01), 
        np.arange(0.10, 0.55, 0.05) 
    ])
    
    best_f1_thresh = 0.5
    max_f1 = 0.0
    
    best_safe_thresh = 0.0
    
    for t in search_space:
        preds = (y_scores >= t).astype(int)
        
        # Confusion Matrix components
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tp = np.sum((preds == 1) & (y_true == 1))
        
        # Metrics
        norm_prec = tn / (tn + fn) if (tn + fn) > 0 else 0 # Safety
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0    # Detection
        tb_prec = tp / (tp + fp) if (tp + fp) > 0 else 0   # False Alarms
        
        # F1 Score
        f1 = 2 * (tb_prec * recall) / (tb_prec + recall) if (tb_prec + recall) > 0 else 0
        
        # Track Best F1
        if f1 > max_f1:
            max_f1 = f1
            best_f1_thresh = t
            
        # Track Best Safety (First threshold where NP >= 0.90)
        if norm_prec >= 0.90 and best_safe_thresh == 0:
            best_safe_thresh = t

        # Print samples for visibility
        if t in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
             print(f"{t:.2f}     | {norm_prec:.4f}     | {recall:.4f}     | {f1:.4f}     | {tb_prec:.4f}")

    print(f"{'='*75}")
    print(f"Best F1 Score ({max_f1:.4f}) found at Threshold: {best_f1_thresh:.2f}")
    
    if best_safe_thresh > 0:
        print(f"Safest Threshold (>0.90 NP) found at: {best_safe_thresh:.2f}")
    else:
        print(f"No threshold reached 0.90 Normal Precision.")
        
    # --- PLOTTING ---
    try:
        t_vals = np.linspace(0.01, 0.99, 100)
        np_curve, f1_curve, rec_curve = [], [], []
        
        for t in t_vals:
            p = (y_scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0,1]).ravel()
            np_curve.append(tn/(tn+fn) if (tn+fn)>0 else 0)
            rec_curve.append(tp/(tp+fn) if (tp+fn)>0 else 0)
            f1 = 2*(tp/(tp+fn) * tp/(tp+fp))/(tp/(tp+fn) + tp/(tp+fp)) if (tp+fn+fp)>0 else 0
            f1_curve.append(f1)

        plt.figure(figsize=(10, 6))
        
        plt.plot(t_vals, np_curve, label='Normal Prec (Safety)', color='green', linewidth=2)
        plt.plot(t_vals, f1_curve, label='F1 Score (Balance)', color='blue', linestyle='--')
        plt.plot(t_vals, rec_curve, label='TB Recall', color='red', linestyle=':')
        plt.axvline(x=best_f1_thresh, color='gray', alpha=0.5, label=f'Best F1 ({best_f1_thresh:.2f})')
        plt.title("Threshold Trade-off Analysis")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig('threshold_analysis.png')
        print("Chart saved as 'threshold_analysis.png'")
    except Exception as e:
        print(f"Could not plot: {e}")

    # --- FINAL SELECTION ---
    # We default to the Best F1 threshold for the report to show the "Balanced" view.
    # Change to 'best_safe_thresh' if you prefer safety.
    final_thresh = best_f1_thresh 
    
    print(f"\nUSING FINAL THRESHOLD: {final_thresh:.4f}")
    preds = (y_scores >= final_thresh).astype(int)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Metrics
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, target_names=class_names, zero_division=0)
    
    print(f"\n{'='*20} FINAL RESULTS {'='*20}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    tn, fp, fn, tp = cm.ravel()
    norm_prec = tn / (tn + fn) if (tn+fn) > 0 else 0
    print(f"\nNormal Precision: {norm_prec:.4f}")

    if save_csv_path:
        df = pd.DataFrame({
            "path": all_paths,
            "true_label": [class_names[i] for i in y_true],
            "tb_prob": y_scores,
            "pred_label": [class_names[i] for i in preds]
        })
        df.to_csv(save_csv_path, index=False)
        print(f"Predictions saved to {save_csv_path}")

# ----------------------------------------
# Main
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-csv", type=str, default="predictions_tuned.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class-names", type=str, nargs="+", default=["Normal", "Tuberculosis"])
    args = parser.parse_args()

    # --- STANDARD PREPROCESSING ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    if args.data_dir:
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    elif args.csv:
        dataset = ImageCSV(args.csv, transform=transform)
    else:
        raise ValueError("Provide --data-dir or --csv")
    
    # FIX: num_workers=0 to prevent Windows Multiprocessing Error
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = build_model(num_classes=len(args.class_names), device=args.device, weights_load_path=args.weights)
    
    evaluate(model, dataloader, args.device, args.class_names, save_csv_path=args.save_csv)

if __name__ == "__main__":
    main()