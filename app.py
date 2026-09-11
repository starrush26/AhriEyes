# cd "C:\Users\kihyu\OneDrive\바탕 화면\Project_AhriEyes"
# .\ahrieyesvenv\Scripts\Activate.ps1
# uvicorn app:app --reload
# python tunnel.py

# Restart Language Server

# git add .
# git commit -m "update"
# git push origin main

import os
import gc
import traceback
import numpy as np
from PIL import Image
import joblib
import onnxruntime as ort
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# 모델 저장 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok = True)

# 릴리즈 다운로드 베이스 URL
RELEASE_URL = "https://github.com/starrush26/AhriEyes/releases/download/v1.0.0"

# 다운로드 대상 파일 목록
MODEL_FILES = [
    "convnext.onnx",
    "convnext.onnx.data",
    "efficientnet.onnx",
    "efficientnet.onnx.data",
    "vit.onnx",
    "vit.onnx.data"
]

# 모델 파일 존재 여부 확인 및 다운로드 함수 정의
def ensure_models_exist():
    for filename in MODEL_FILES:
        file_path = os.path.join(MODEL_DIR, filename)

        # 모델 파일 존재 여부 확인
        if not os.path.exists(file_path):
            download_url = f"{RELEASE_URL}/{filename}"
            print(f"[Model Downloader] {filename} 다운로드 중...")
            urllib.request.urlretrieve(download_url, file_path)
            print(f"[Model Downloader] {filename} 완료!")

# 서버 부팅 시 모델 파일 존재 여부 확인 및 자동 다운로드
ensure_models_exist()

# -------------------------------------------------------------
# [서버 부팅 시 사전 웜업(Warm-up) 수명주기 정의]
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n [AhriEyes] 서버 기동 중: 3중 ONNX 엔진 사전 웜업을 시작합니다...")
    try:
        # 웜업을 위한 가짜 더미 입력 데이터 생성 (1, 3, 224, 224)
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        
        # 1. EfficientNet 웜업
        sess_eff = get_onnx_session("efficientnet.onnx")
        sess_eff.run(None, {sess_eff.get_inputs()[0].name: dummy_input})
        del sess_eff
        
        # 2. ConvNeXt 웜업
        sess_conv = get_onnx_session("convnext.onnx")
        sess_conv.run(None, {sess_conv.get_inputs()[0].name: dummy_input})
        del sess_conv
        
        # 3. ViT 웜업
        sess_vit = get_onnx_session("vit.onnx")
        sess_vit.run(None, {sess_vit.get_inputs()[0].name: dummy_input})
        del sess_vit
        
        # 4. 메타 모델 웜업
        meta_path = os.path.join(MODELS_DIR, "stacking_meta_logistic_model.pkl")
        meta_model = joblib.load(meta_path)
        meta_model.predict_proba(np.array([[0.5, 0.5, 0.5]], dtype=np.float32))
        del meta_model
        
        gc.collect()
        print("[AhriEyes] 3중 앙상블 모델 메모리 캐싱 및 웜업 완료! \n")
    except Exception as e:
        print(f" [AhriEyes] 웜업 중 오류 발생 (실제 판독 시 로드됨): {e}")

    yield  # 서버 실행 중 파이프라인

    print("\n [AhriEyes] 서버가 종료됩니다.")

# -------------------------------------------------------------
# [환경 설정 및 경로 초기화]
# -------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
SOUND_DIR = os.path.join(BASE_DIR, "static", "sound")

# FastAPI 앱 생성 시 lifespan 등록
app = FastAPI(title="AhriEyes 딥페이크 탐지 시스템", lifespan=lifespan)
app.mount("/sound", StaticFiles(directory=SOUND_DIR), name="sound")

# 정적 파일 및 템플릿 연결
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/sound", StaticFiles(directory=SOUND_DIR), name="sound")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -------------------------------------------------------------
# [유틸리티 함수: 전처리 및 소프트맥스]
# -------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    PIL과 NumPy 기반 전처리 파이프라인
    - Bicubic 보간법 리사이징 (224x224)
    - ImageNet 정규화 (Mean / Std)
    - (1, 3, 224, 224) C-연속 메모리 float32 반환
    """
    # 1. 224x224 Bicubic 보간 리사이즈
    resized = image.resize((224, 224), resample=Image.Resampling.BICUBIC)
    
    # 2. [0, 1] 범위 정규화 (H, W, C)
    arr = np.array(resized, dtype=np.float32) / 255.0

    # 3. ImageNet 기준 표준화
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # 4. 차원 변경: (H, W, C) -> (C, H, W) -> 배치 차원 추가 (1, C, H, W)
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    # ONNX C++ 네이티브 입력용 연속 메모리 보장
    return np.ascontiguousarray(arr, dtype=np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    """수치적 안정성을 고려한 소프트맥스"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def get_onnx_session(filename: str) -> ort.InferenceSession:
    """
    로컬 models 폴더 내 파일 경로 매핑
    .onnx와 .onnx.data가 나란히 위치해야 외부 가중치를 자동 로드
    """
    model_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"오류 : 모델 파일을 찾을 수 없습니다: {model_path}")
    
    providers = ['CPUExecutionProvider']
    return ort.InferenceSession(model_path, providers = providers)

# -------------------------------------------------------------
# [엔드포인트 라우팅]
# -------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    """메인 대시보드 페이지 렌더링"""
    return templates.TemplateResponse(
        request = request, 
        name = "index.html"
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """3대 앙상블 ONNX 추론 및 메타 로지스틱 회귀 판독 파이프라인"""
    try:
        # 1. 업로드 이미지 로드 및 전처리
        image = Image.open(file.file).convert("RGB")
        input_data = preprocess_image(image)

        # --- [1단계] EfficientNet ONNX 추론 ---
        session_eff = get_onnx_session("efficientnet.onnx")
        input_name_eff = session_eff.get_inputs()[0].name
        out_eff = session_eff.run(None, {input_name_eff: input_data})[0]
        prob_eff = float(softmax(out_eff)[0][0])  # 가짜(Fake) 클래스 인덱스 확률

        del session_eff, out_eff # 메모리 해제

        # --- [2단계] ConvNeXt ONNX 추론 ---
        session_conv = get_onnx_session("convnext.onnx")
        input_name_conv = session_conv.get_inputs()[0].name
        out_conv = session_conv.run(None, {input_name_conv: input_data})[0]
        prob_conv = float(softmax(out_conv)[0][0])

        del session_conv, out_conv

        # --- [3단계] ViT ONNX 추론 ---
        session_vit = get_onnx_session("vit.onnx")
        input_name_vit = session_vit.get_inputs()[0].name
        out_vit = session_vit.run(None, {input_name_vit: input_data})[0]
        prob_vit = float(softmax(out_vit)[0][0])

        del session_vit, out_vit, input_data
        gc.collect()  # 대형 모델 추론 후 힙 메모리 1회 집중 정리

        # --- [4단계] Meta Stacking 모델 판별 ---
        meta_path = os.path.join(MODELS_DIR, "stacking_meta_logistic_model.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"메타 모델을 찾을 수 없습니다: {meta_path}")
        
        meta_model = joblib.load(meta_path)
        features = np.array([[prob_eff, prob_conv, prob_vit]], dtype=np.float32)
        # 메타 모델의 클래스 1(가짜) 확률 도출
        final_prob = float(meta_model.predict_proba(features)[0][1] * 100.0)

        del meta_model
        gc.collect()

        # 판독 라벨 및 신뢰도 산출
        label = "FAKE (AI 생성)" if final_prob >= 50.0 else "REAL (실제 사진)"
        confidence = final_prob if final_prob >= 50.0 else (100.0 - final_prob)

        # 프론트엔드 규격에 완벽히 맞춘 반환 데이터
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
        traceback.print_exc()
        return {"error": str(e), "label": None}