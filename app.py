# cd "C:\Users\kihyu\OneDrive\바탕 화면\Project_AhriEyes"
# .\ahrieyesvenv\Scripts\Activate.ps1
# uvicorn app:app --reload
# python tunnel.py

# Restart Language Server

# git add .
# git commit -m "update to onnxruntime"
# git push origin main

import os
import gc
from pathlib import Path
from PIL import Image
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

app = FastAPI(title="AhriEyes Deepfake Detector")

# /static, /sound 요청을 실제 폴더로 연결
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/sound", StaticFiles(directory="static/sound"), name="sound")

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "templates" / "index.html"

HF_REPO_ID = "kihyeonlee/AhriEyes-weights"

def load_file_from_hf(filename: str):
    """허깅페이스에서 모델 파일 캐시 다운로드 (.onnx의 경우 .data 파일도 함께 확인)"""
    if filename.endswith(".onnx"):
        data_filename = f"{filename}.data"
        try:
            hf_hub_download(repo_id=HF_REPO_ID, filename=data_filename)
        except Exception:
            pass  # 데이터 파일이 분리되지 않은 단일 onnx일 경우 무시
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """순수 NumPy / PIL 기반 이미지 정규화 전처리 (Torchvision 대체)"""
    # 1. 224x224 리사이즈 (ViT에 최적화된 쌍입방(Bicubic) 보간법)
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    
    # 2. 0~1 정규화 및 float32 변환
    img_data = np.array(image, dtype=np.float32) / 255.0
    
    # 3. ImageNet 표준 정규화
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # 4. HWC -> CHW 변환 및 배치 차원 추가: (1, 3, 224, 224)
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)

def softmax(x: np.ndarray) -> np.ndarray:
    """NumPy 기반 소프트맥스 함수"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse(HTML_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. 이미지 로드 및 전처리 (NumPy 배열로 즉시 준비)
        image = Image.open(file.file).convert("RGB")
        input_data = preprocess_image(image)

        # CPU 연산 최적화 세션 옵션 (가벼운 메모리 할당)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # --- [1단계] EfficientNet ONNX 추론 ---
        eff_path = load_file_from_hf("efficientnet.onnx")
        session_eff = ort.InferenceSession(eff_path, sess_options, providers=['CPUExecutionProvider'])
        input_name_eff = session_eff.get_inputs()[0].name
        
        out_eff = session_eff.run(None, {input_name_eff: input_data})[0]
        # 0번 인덱스가 FAKE 확률
        prob_eff = float(softmax(out_eff)[0][0])
        
        del session_eff, out_eff # 메모리 해제
        gc.collect() # 가비지 컬렉션 호출

        # --- [2단계] ConvNeXt ONNX 추론 ---
        conv_path = load_file_from_hf("convnext.onnx")
        session_conv = ort.InferenceSession(conv_path, sess_options, providers=['CPUExecutionProvider'])
        input_name_conv = session_conv.get_inputs()[0].name
        
        out_conv = session_conv.run(None, {input_name_conv: input_data})[0]
        # 0번 인덱스가 FAKE 확률
        prob_conv = float(softmax(out_conv)[0][0])
        
        del session_conv, out_conv
        gc.collect()

        # --- [3단계] ViT ONNX 추론 ---
        vit_path = load_file_from_hf("vit.onnx")
        session_vit = ort.InferenceSession(vit_path, sess_options, providers=['CPUExecutionProvider'])
        input_name_vit = session_vit.get_inputs()[0].name
        
        out_vit = session_vit.run(None, {input_name_vit: input_data})[0]
        # 0번 인덱스가 FAKE 확률
        prob_vit = float(softmax(out_vit)[0][0])
        
        del session_vit, out_vit, input_data
        gc.collect()

        # --- [4단계] Meta Logistic Regression 앙상블 판정 ---
        meta_path = load_file_from_hf("stacking_meta_logistic_model.pkl")
        meta_model = joblib.load(meta_path)
        
        features = np.array([[prob_eff, prob_conv, prob_vit]])
        # 메타 모델은 1번 인덱스가 FAKE 확률
        final_prob = float(meta_model.predict_proba(features)[0][1] * 100.0)

        del meta_model
        gc.collect()

        # 라벨 및 신뢰도 판정
        label = "FAKE (AI 생성)" if final_prob >= 50.0 else "REAL (실제 사진)"
        confidence = final_prob if final_prob >= 50.0 else (100.0 - final_prob)

        # 추론 결과를 JSON 형태로 반환
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
    # 오류 발생 시 예외처리
    except Exception as e:
        return {"error": str(e)}