from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI()

# --- 1. KONFIGURASI CORS (Agar React bisa akses API ini) ---
origins = [
    "http://localhost:3000",  # Port default React
    "http://localhost:5173",  # Port default Vite
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Untuk development boleh "*" (semua)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet50(weights=None)
    # Sesuaikan head dengan training kita
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    # Load bobot
    try:
        model.load_state_dict(torch.load("tb_resnet50_robust.pth", map_location=device))
        print("Model berhasil dimuat!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    model.eval()
    return model

model = load_model()

# --- 3. PREPROCESSING IMAGE ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. ENDPOINT API ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Baca file gambar
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Proses ke Tensor
    img_tensor = transform(image).unsqueeze(0) # Tambah batch dimension
    
    # Prediksi
    with torch.no_grad():
        outputs = model(img_tensor)
        # Hitung persentase
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
    
    class_names = ['Normal', 'Tuberculosis']
    result = class_names[preds.item()]
    confidence = conf.item() * 100
    
    return {
        "filename": file.filename,
        "prediction": result,
        "confidence": f"{confidence:.2f}%",
        "is_tb": result == 'Tuberculosis'
    }