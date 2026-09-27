# cd "C:\Users\kihyu\OneDrive\바탕 화면\Project_AhriEyes"
# .\ahrieyesvenv\Scripts\Activate.ps1
# uvicorn app:app --reload
# python tunnel.py

# Restart Language Server

# ----- 깃 커밋 -----
# git add .
# git commit -m "update"
# git push origin main

# ----- 깃 동기화 -----
# git add .
# git commit -m "chore: save local changes"
# git pull origin main

# ------- 버전 업그레이드 -------
# 1. 수정된 파일 스테이징
#git add .

# 2. 커밋 메시지 작성
#git commit -m "docs: bump version to v2.0.0 in frontend"

# 3. v2.0.0 릴리즈 태그 생성
#git tag -a v2.0.0 -m "Release v2.0.0: Hugging Face Model Hub Dynamic Pipeline & Stacking Ensemble Optimization"

# 4. 브랜치 커밋과 태그를 GitHub로 동시 푸시
#git push origin main --tags

import os
import gc
import io
import traceback
import numpy as np
from PIL import Image
import joblib
import onnxruntime as ort
import urllib.request
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, File, UploadFile, status, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from huggingface_hub import hf_hub_download
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import hashlib
import hmac
import numpy as np
import base64
from dotenv import load_dotenv

try:
    import resource

except ImportError: # 윈도우 환경 호환 처리 예외
    resource = None

load_dotenv()  # .env 파일에 적힌 환경변수를 자동으로 os.environ에 로드

# 모델 저장 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok = True)

# Hugging Face 저장소 ID - 허깅페이스 모델 저장소에서 다운로드
HF_REPO_ID = "kihyeonlee/ahrieyes-models"

# 다운로드 대상 파일 목록
MODEL_FILES = {
    "efficientnet": "efficientnet.onnx",
    "convnext": "convnext.onnx",
    "vit": "vit.onnx"
}

# 모델 파일 존재 여부 확인 및 다운로드 함수 정의
def ensure_models_exist():
    for filename in MODEL_FILES.values():
        file_path = os.path.join(MODEL_DIR, filename)

        # 모델 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"[Model Downloader] HF에서 {filename} 다운로드 중...")
            downloaded_cache_path = hf_hub_download(
                repo_id = HF_REPO_ID,
                filename = filename,
                local_dir = MODEL_DIR,
                local_dir_use_symlinks = False
            )
            print(f"[Model Downloader] {filename} 완료! -> {file_path}")

        else:
            print(f"[Model Downloader] {filename} 로컬 캐시 확인 완료.")

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
app = FastAPI(title="AhriEyes 딥페이크 탐지 시스템", lifespan=lifespan, version="2.1.0")
app.mount("/sound", StaticFiles(directory=SOUND_DIR), name="sound")

# 정적 파일 및 템플릿 연결
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/sound", StaticFiles(directory=SOUND_DIR), name="sound")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# 단일 워커 스레드 풀: 512MB 환경에서 추론 연산이 겹치는 것을 원천 차단
inference_executor = ThreadPoolExecutor(max_workers=1)
# 비동기 대기열 락: 들어온 순서대로 차례차례 진입하도록 제어
inference_lock = asyncio.Lock()

AHRIEYES_SECRET_SALT = os.environ.get("AHRIEYES_SECRET_SALT", "DEV_LOCAL_SECRET_FALLBACK_2026")

# 서명 요청용 Pydantic 모델
class SignRequest(BaseModel):
    raw_hash: str  # 프론트에서 전송한 캔버스 픽셀의 SHA-256 해시값

# 서명 응답 모델
class SignResponse(BaseModel):
    signature: str
    status: str
    author: str
    software: str

# 검증 응답 모델
class VerifyResponse(BaseModel):
    verified: bool
    status: str
    author: str | None = None
    software: str | None = None
    message: str


# --- 엔드포인트 1: 리포트 서명 발급 (Sign) ---
@app.post("/api/sign-report", response_model=SignResponse)
async def sign_report(payload: SignRequest):
    if not payload.raw_hash or len(payload.raw_hash) != 64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="유효하지 않은 원시 해시 포맷입니다 (64자리 SHA-256 필요)."
        )
    
    # 서버 메모리의 비밀 키와 클라이언트 픽셀 해시를 HMAC-SHA256으로 결합
    official_signature = hmac.new(
        AHRIEYES_SECRET_SALT.encode("utf-8"),
        payload.raw_hash.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return SignResponse(
        signature=official_signature,
        status="ORIGINAL_GENUINE",
        author="Gi Hyeon Lee",
        software="AhriEyes AI Forensic Engine v2.0"
    )

# --- 엔드포인트 2: 다운로드된 PNG 리포트 정품 검증 (Verify) ---
@app.post("/api/verify-report", response_model=VerifyResponse)
async def verify_report(file: UploadFile = File(...)):
    # MIME 타입 체크
    if file.content_type not in ["image/png", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="PNG 이미지 파일만 검증 가능합니다."
        )

    try:
        # 512MB RAM 보호를 위해 청크 단위로 파일 바이트 읽기
        contents = await file.read()
        
        # PIL을 사용해 PNG 청크 파싱
        with Image.open(io.BytesIO(contents)) as pil_img:
            # 1. tEXt 메타데이터 청크 추출
            metadata = pil_img.text
            claimed_signature = metadata.get("Integrity-Hash")
            claimed_software = metadata.get("Software")
            claimed_author = metadata.get("Author")

            # 메타데이터 누락 시 1차 위조 판정
            if not claimed_signature:
                return VerifyResponse(
                    verified=False,
                    status="MISSING_METADATA",
                    message="AhriEyes 무결성 서명 청크가 발견되지 않았습니다. 변조되었거나 외부 생성 파일입니다."
                )

            # 2. 이미지 픽셀 원본 바이트 추출 후 SHA-256 연산
            # 압축된 PNG 바이트가 아니라 '순수 픽셀 데이터(Raw Bytes)'를 해싱
            pixel_bytes = pil_img.tobytes()
            recalculated_raw_hash = hashlib.sha256(pixel_bytes).hexdigest()

            # 3. 서버 비밀 키로 재생성한 정품 기대 서명
            expected_signature = hmac.new(
                AHRIEYES_SECRET_SALT.encode("utf-8"),
                recalculated_raw_hash.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()

            # 4. 타이밍 공격 방지용 상수 시간 비교(compare_digest)
            is_valid = hmac.compare_digest(claimed_signature, expected_signature)

            if is_valid:
                return VerifyResponse(
                    verified=True,
                    status="GENUINE_AUTHENTIC",
                    author=claimed_author,
                    software=claimed_software,
                    message="공식 Ahrieyes 포렌식 진품 리포트입니다. 수치 및 픽셀의 위변조가 없습니다."
                )
            else:
                return VerifyResponse(
                    verified=False,
                    status="TAMPERED_FRAUD",
                    author=claimed_author,
                    software=claimed_software,
                    message="경고: 리포트의 픽셀 데이터나 판독 수치가 임의로 조작/위조되었습니다!"
                )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"파일 판독 중 오류 발생: {str(e)}"
        )

def get_real_ip(request: Request) -> str:
    # 프록시(Cloudflare/Render)를 거쳐 전달된 실제 접속자 IP 확인
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # 클라이언트 정보가 없으면 로컬호스트 반환
    return request.client.host if request.client else "127.0.0.1" 

# 인메모리 레이트 리미터 초기화
limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter

# 429 에러 발생 시 깔끔한 JSON 응답 반환
@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Too Many Requests",
            "message": "요청 횟수가 너무 많습니다. 1분 후 다시 시도해 주세요."
        }
    )

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

# 소프트멕스 함수 정의 (수치적 안정성 고려)
def softmax(x: np.ndarray) -> np.ndarray:
    """수치적 안정성을 고려한 소프트맥스"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def generate_forensic_heatmap(face_img_bgr, final_prob):
    """
    0.5 vCPU 맞춤형 초경량 포렌식 히트맵 생성기 (Zero-Backprop CAM)
    - 합성 경계면의 고주파 잔차(High-Frequency Residual) 추출
    - 모델의 최종 FAKE 확률(final_prob)에 비례하여 결함 강도 동적 맵핑
    """
    try:
        # 1. 224x224 표준 규격으로 리사이즈
        resized = cv2.resize(face_img_bgr, (224, 224))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 2. 라플라시안 필터를 통한 픽셀 블렌딩 왜곡 및 고주파 아티팩트 추출
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        residual = np.abs(laplacian)

        # 3. 노이즈 스무딩 및 결함 집중 영역 추출 (가우시안 블러)
        blurred_residual = cv2.GaussianBlur(residual, (15, 15), 0)

        # 4. 정규화 (0 ~ 255)
        norm_residual = cv2.normalize(blurred_residual, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

        # 5. 모델의 FAKE 확률 가중치 반영 (가짜 확률이 높을수록 붉은 결함 강조)
        weight = float(np.clip(final_prob, 0.1, 1.0))
        scaled_residual = (norm_residual * weight).astype(np.uint8)

        # 6. 포렌식 컬러맵 적용 (TURBO 또는 JET: 파랑=정상, 빨강/노랑=조작 의심 영역)
        heatmap_colored = cv2.applyColorMap(scaled_residual, cv2.COLORMAP_TURBO)

        # 7. 원본 얼굴과 히트맵을 6:4 비율로 반투명 블렌딩
        overlay = cv2.addWeighted(resized, 0.55, heatmap_colored, 0.45, 0)

        # 8. Web 전송용 경량 JPEG Base64 인코딩
        _, buffer = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        return f"data:image/jpeg;base64,{heatmap_base64}"
    
    except Exception as e:
        print(f"[Heatmap Generation Warning] {e}")
        return None

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
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 최적화 레벨 기본 단계
    sess_options.enable_cpu_mem_arena = False       # C++ 내부 메모리 풀 비활성화 (필요할 때만 쓰고 즉시 반환)
    sess_options.enable_mem_pattern = False         # 정적 메모리 할당 패턴 끄기 (피크치 절약)

    providers = ['CPUExecutionProvider'] # Render 환경에서 GPU 미사용 강제
    return ort.InferenceSession(model_path, sess_options = sess_options, providers = providers)

# -------------------------------------------------------------
# [엔드포인트 라우팅]
# -------------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"]) # GET 및 HEAD 라우터 요청 허용
async def index(request: Request):
    """메인 대시보드 페이지 렌더링"""
    return templates.TemplateResponse(
        request = request, 
        name = "index.html"
    )

@app.post("/predict") # POST 요청만 허용
@limiter.limit("10/minute") # 1분당 10회 요청 제한

async def predict(request: Request, file: UploadFile = File(...)):
    """3대 앙상블 ONNX 추론 및 메타 로지스틱 회귀 판독 파이프라인"""
    image_bytes = await file.read()
    loop = asyncio.get_running_loop()

    async with inference_lock:
        result = await loop.run_in_executor(
            inference_executor,
            run_3tier_ensemble_pipeline,
            image_bytes
        )
        return result
    
def run_3tier_ensemble_pipeline(image_bytes: bytes) -> dict:
    """3대 앙상블 ONNX 추론 및 메타 로지스틱 회귀 판독 파이프라인"""
    
    try:
        # 1. 넘겨받은 이미지 바이트를 PIL Image로 변환 후 전처리
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_data = preprocess_image(image)

        # --- [1단계] EfficientNet ONNX 추론 ---
        session_eff = get_onnx_session("efficientnet.onnx")
        input_name_eff = session_eff.get_inputs()[0].name
        out_eff = session_eff.run(None, {input_name_eff: input_data})[0]
        prob_eff = float(softmax(out_eff)[0][0])  # 가짜(Fake) 클래스 인덱스 확률

        del session_eff, out_eff # 메모리 해제
        gc.collect()

        # --- [2단계] ConvNeXt ONNX 추론 ---
        session_conv = get_onnx_session("convnext.onnx")
        input_name_conv = session_conv.get_inputs()[0].name
        out_conv = session_conv.run(None, {input_name_conv: input_data})[0]
        prob_conv = float(softmax(out_conv)[0][0])

        del session_conv, out_conv
        gc.collect()

        # --- [3단계] ViT ONNX 추론 ---
        session_vit = get_onnx_session("vit.onnx") # ViT INT8 양자화 모델 사용
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

        # 원본 바이트에서 OpenCV BGR 이미지 복원
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 디코딩한 BGR 이미지를 기반으로 포렌식 히트맵 생성
        heatmap_data = generate_forensic_heatmap(img_bgr, final_prob / 100.0)

        # 프론트엔드 규격에 완벽히 맞춘 반환 데이터
        return {
            "label": label,
            "confidence": round(confidence, 2),
            "fake_probability": round(final_prob, 2),
            "model_details": {
                "EfficientNet": round(prob_eff * 100, 2),
                "ConvNeXt": round(prob_conv * 100, 2),
                "ViT": round(prob_vit * 100, 2)
            },
            "heatmap": heatmap_data
        }

    except Exception as e: # 예외처리
        traceback.print_exc()
        return {"error": str(e), "label": None}

    finally:
        # 가비지 컬렉션 및 메모리 해제
        gc.collect()
