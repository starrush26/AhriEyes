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
import resource

# 모델 저장 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok = True)

# 릴리즈 다운로드 베이스 URL
RELEASE_URL = "https://github.com/starrush26/AhriEyes/releases/download/v1.0.0"

# 다운로드 대상 파일 목록
MODEL_FILES = [
    "efficientnet_int8.onnx",
    "convnext_int8.onnx",
    "vit_int8.onnx",
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
    print("\n[AhriEyes] 서버 기동 완료 (메모리 절약 모드 가동)")

    yield

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
    # 1. 224x224 BILINEAR (쌍선형 2*2참조) 보간 리사이즈
    resized = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
    
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

# Render 초소형 vCPU 맞춤 경량화 옵션
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1  # 단일 연산 내부 스레드 1개 강제
opts.inter_op_num_threads = 1  # 연산 간 병렬 스레드 1개 강제
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # 순서 실행 모드 강제
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 최적화 레벨 최대화

#오닉스 세션 로딩 함수
def get_onnx_session(filename: str) -> ort.InferenceSession:
    """
    로컬 models 폴더 내 파일 경로 매핑
    - 모델 파일들이 나란히 위치해야 외부 가중치를 자동 로드
    """
    model_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"오류 : 모델 파일을 찾을 수 없습니다: {model_path}")

    # Render 저스펙 CPU 병목 방지 옵션
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1 # 단일 연산 내부 스레드 1개 강제
    sess_options.inter_op_num_threads = 1 # 연산 간 병렬 스레드 1개 강제
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # 순서 실행 모드 강제
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 최적화 레벨 최대화
    sess_options.enable_cpu_mem_arena = False       # C++ 내부 메모리 풀 비활성화 (필요할 때만 쓰고 즉시 반환)
    sess_options.enable_mem_pattern = False         # 정적 메모리 할당 패턴 끄기 (피크치 절약)

    providers = ['CPUExecutionProvider'] # Render 환경에서 GPU 미사용 강제
    return ort.InferenceSession(model_path, sess_options = sess_options, providers = providers)

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
        session_eff = get_onnx_session("efficientnet_int8.onnx")
        input_name_eff = session_eff.get_inputs()[0].name
        out_eff = session_eff.run(None, {input_name_eff: input_data})[0]
        prob_eff = float(softmax(out_eff)[0][0])  # 가짜(Fake) 클래스 인덱스 확률

        del session_eff, out_eff # 메모리 해제
        gc.collect()

        # --- [2단계] ConvNeXt ONNX 추론 ---
        session_conv = get_onnx_session("convnext_int8.onnx")
        input_name_conv = session_conv.get_inputs()[0].name
        out_conv = session_conv.run(None, {input_name_conv: input_data})[0]
        prob_conv = float(softmax(out_conv)[0][0])

        del session_conv, out_conv
        gc.collect()

        # --- [3단계] ViT ONNX 추론 ---
        session_vit = get_onnx_session("vit_int8.onnx") # ViT INT8 양자화 모델 사용
        input_name_vit = session_vit.get_inputs()[0].name
        out_vit = session_vit.run(None, {input_name_vit: input_data})[0]
        prob_vit = float(softmax(out_vit)[0][0])

        del session_vit, out_vit, input_data
        gc.collect()  

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

        # 추론 완료 후 최종 메모리 피크 확인 (Linux 환경 전용, Windows 예외 처리)
        try:
            max_ram_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"\n[AhriEyes Memory Check] 프로세스 최고 메모리 피크: {max_ram_used:.2f} MB / 512.00 MB\n")

        except (ImportError, AttributeError):
            pass

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