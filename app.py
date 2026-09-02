# cd "C:\Users\kihyu\OneDrive\바탕 화면\Project_AhriEyes"
# .\ahrieyesvenv\Scripts\Activate.ps1
# uvicorn app:app --reload
# python tunnel.py

# git add .
# git commit -m "update"
# git push origin main
import os
import gc
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from huggingface_hub import hf_hub_download

app = FastAPI(title="AhriEyes Deepfake Detector")

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "templates" / "index.html"

HF_REPO_ID = "kihyeonlee/AhriEyes-weights"

# 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_file_from_hf(filename: str):
    """허깅페이스에서 모델 파일 캐시 다운로드"""
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)

@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(HTML_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. 이미지 로드 및 전처리
        image = Image.open(file.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        # --- [1단계] EfficientNet 추론 및 메모리 해제 ---
        eff_path = load_file_from_hf("EfficientNet_deepfake.pth")
        m_eff = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        m_eff.load_state_dict(torch.load(eff_path, map_location="cpu"))
        m_eff.eval()
        
        with torch.no_grad():
            out_eff = m_eff(input_tensor)
            # 0번 인덱스가 FAKE 확률
            prob_eff = torch.softmax(out_eff, dim=1)[0][0].item()
            
        del m_eff, out_eff # 메모리 해제
        gc.collect() # 가비지 컬렉션 호출

        # --- [2단계] ConvNeXt 추론 및 메모리 해제 ---
        conv_path = load_file_from_hf("ConvNeXt_deepfake.pth")
        m_conv = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
        m_conv.load_state_dict(torch.load(conv_path, map_location="cpu"))
        m_conv.eval()
        
        with torch.no_grad():
            out_conv = m_conv(input_tensor)
            # 0번 인덱스가 FAKE 확률
            prob_conv = torch.softmax(out_conv, dim=1)[0][0].item()
            
        del m_conv, out_conv
        gc.collect()

        # --- [3단계] ViT 추론 및 메모리 해제 ---
        vit_path = load_file_from_hf("ViT_deepfake.pth")
        m_vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
        m_vit.load_state_dict(torch.load(vit_path, map_location="cpu"))
        m_vit.eval()
        
        with torch.no_grad():
            out_vit = m_vit(input_tensor)
            # 0번 인덱스가 FAKE 확률
            prob_vit = torch.softmax(out_vit, dim=1)[0][0].item()
            
        del m_vit, out_vit, input_tensor
        gc.collect()

        # --- [4단계] Meta Logistic Regression 앙상블 판정 ---
        meta_path = load_file_from_hf("stacking_meta_logistic_model.pkl")
        meta_model = joblib.load(meta_path)
        
        features = np.array([[prob_eff, prob_conv, prob_vit]])
        # 메터 모델은 1번 인덱스가 FAKE 확률
        final_prob = meta_model.predict_proba(features)[0][1] * 100.0

        del meta_model
        gc.collect()

        # 라벨 및 신뢰도 판정
        label = "FAKE (AI 생성)" if final_prob >= 50.0 else "REAL (실제 사진)"
        confidence = final_prob if final_prob >= 50.0 else (100.0 - final_prob)

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "fake_probability": round(final_prob, 2),
            "model_details": {
                "EfficientNet": round(prob_eff * 100, 2),
                "ConvNeXt": round(prob_conv * 100, 2),
                "ViT": round(prob_vit * 100, 2)
            }
        }

    except Exception as e:
        return {"error": str(e)}