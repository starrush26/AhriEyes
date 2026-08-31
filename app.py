# cd "C:\Users\kihyu\OneDrive\바탕 화면\Project_AhriEyes"
# uvicorn app:app --reload
# python tunnel.py

import os
import io
import gc
import traceback
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import hf_hub_download

app = FastAPI()

templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

REPO_ID = "kihyeonlee/AhriEyes-weights"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cpu')
torch.set_num_threads(1)

def get_model_path(filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[HF 다운로드 시작] {filename}...")
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODEL_DIR)
        print(f"[HF 다운로드 완료] {filename}")
    return path

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_and_predict_single(model_creator, pth_filename: str, img_tensor: torch.Tensor) -> float:
    gc.collect()
    pth_path = get_model_path(pth_filename)
    
    model = model_creator()
    state_dict = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    with torch.no_grad():
        out = model(img_tensor)
        # 딥러닝 백본 모델은 0번 인덱스가 Fake
        prob = torch.softmax(out, dim=1)[0, 0].item()
        
    del model
    del state_dict
    gc.collect()
    return prob

def get_eff_model():
    return timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

def get_conv_model():
    return timm.create_model('convnext_tiny', pretrained=False, num_classes=2)

def get_vit_model():
    return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print("\n===== [측정 시작] =====")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        # 1. 3개 모델 개별 Fake 확률 추출 (0번 인덱스)
        p_eff = load_and_predict_single(get_eff_model, 'EfficientNet_deepfake.pth', img_tensor)
        p_conv = load_and_predict_single(get_conv_model, 'ConvNeXt_deepfake.pth', img_tensor)
        p_vit = load_and_predict_single(get_vit_model, 'ViT_deepfake.pth', img_tensor)

        print(f"-> 개별 추론 결과: Eff={p_eff:.4f}, Conv={p_conv:.4f}, ViT={p_vit:.4f}")

        # 2. 메타 스태킹 로지스틱 모델 (메타 모델은 1번 인덱스가 Fake)
        meta_path = get_model_path('stacking_meta_logistic_model.pkl')
        meta = joblib.load(meta_path)
        meta_probs = meta.predict_proba([[p_eff, p_conv, p_vit]])[0]
        
        # 메타 모델의 1번 인덱스(Fake) 선택
        final_prob = meta_probs[1] if len(meta_probs) > 1 else meta_probs[0]
        print(f"-> 최종 Fake 판정 확률: {final_prob:.4f}")

        del meta
        gc.collect()

        final_pct = round(final_prob * 100, 1)
        eff_pct = round(p_eff * 100, 1)
        conv_pct = round(p_conv * 100, 1)
        vit_pct = round(p_vit * 100, 1)

        if final_pct < 20: stage = "STAGE 1: 원본 (안전)"
        elif final_pct < 40: stage = "STAGE 2: 의심 (주의)"
        elif final_pct < 60: stage = "STAGE 3: 위험 (경고)"
        elif final_pct < 80: stage = "STAGE 4: 고위험 (심각)"
        else: stage = "STAGE 5: 딥페이크 (위험)"

        print(f"-> 최종 반환: {final_pct}% ({stage})")
        print("===== [측정 완료] =====\n")

        return JSONResponse({
            "status": "success",
            "final_score": final_pct,
            "stage": stage,
            "details": {
                "efficientnet": eff_pct,
                "convnext": conv_pct,
                "vit": vit_pct
            }
        })
    except Exception as e:
        print("\n[백엔드 ERROR 발생]")
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)