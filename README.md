# 🦊 AhriEyes: Real-Time Deepfake Detection System

> **3종 이종 스태킹 앙상블 기반 초경량 딥페이크 탐지 웹 서비스**  
> 라이엇 게임 리그오브 레전드(League of Legends) 영혼의 꽃 아리(Spirit Blossom Ahri)' 콘셉트 UI & 사운드 이펙트 적용

---

## 📌 Project Overview
- **목적**: 무분별한 딥페이크 위변조 이미지로부터 디지털 신뢰성을 검증하기 위한 비영리 공익 프로젝트
- **특징**: 고성능 3종 딥러닝 백본 앙상블을 0.5 vCPU, 512MB RAM 극저비용 무료 서버 환경에 최적화하여 실시간 서빙 구현

---

## 🛠 Tech Stack & Architecture

### Model Architecture (Ensemble)
* **EfficientNet**: 미세 텍스처 및 얼굴 경계면 아티팩트 검출
* **ConvNeXt**: 합성 시 발생하는 구조적 왜곡 및 국소 왜곡 포착
* **Vision Transformer (ViT)**: 전역적 패치 간 어텐션 기반 부자연스러움 탐지
* **Logistic Regression**: 세 개의 모델의 결과를 받아 가중치를 매겨 0~1 사이의 값으로 결과 반환
* **Inference Engine**: ONNX Runtime 기반 INT8 동적 양자화 기반 최적화 서빙(EfficientNet 제외), FP16 변환

### Infrastructure & Backend
* **Language**: Python 3.10.11
* **Backend**: Flask / PyTorch / ONNX Runtime
* **Deployment**: Render (Free Tier: 0.5 vCPU, 512MB RAM, CPU-only / No GPU)

---

## 🚀 Key Achievements
* **극단적 리소스 최적화**: OOM(Out of Memory) 방지를 위해 가중치 최적화 및 램 점유 최소화 파이프라인 구축, 각 모델 로드 종료 후 판독 결과만 뽑고 del과 gc.collect()로 메모리 리셋
* **인터랙티브 UI**: 판독 진행 중, 판독 완료 후 위변조 확률에 따른 5단계 동적 게이지 및 테마 사운드 효과 제공

---

## 📜 Copyright & Citation
본 프로젝트는 공익 및 연구 목적으로 제작된 오픈소스 소프트웨어입니다.

* **Developer**: [KOR] 이기현 (Lee Ki-hyun)
  * **GitHub**: [github.com/starrush26](https://github.com/starrush26)
  * **Kaggle**: [kaggle.com/amongasu](https://www.kaggle.com/amongasu)

* **Dataset Acknowledgement**: Manjil Karki [NPL] (Kaggle Dataset Creator)
  * **Dataset**: [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
  * **GitHub**: [github.com/Manjil-Karki](https://github.com/Manjil-Karki)
  * **Kaggle**: [kaggle.com/manjilkarki](https://www.kaggle.com/manjilkarki)

* **First Release**: 2026.09.11
* **License**: GNU Affero General Public License v3.0
* **© 2026 Lee gi Hyeon. All rights reserved. (Licensed under AGPL-3.0)**
* **Dataset courtesy of Manjil Karki (Deepfake and Real Images Dataset).**

> 본 리포지토리의 코드를 인용, 변형, 연구, 논문 작성, 파생 프로젝트, 대회 및 공모전 출품, 언론 보도로 사용할 경우 반드시 **원작자(이기현) 표기 및 동일 라이선스(AGPL-3.0)** 규정을 준수해야 합니다.
> 무단 도용 및 허위 저작권 주장은 엄격히 금지됩니다.
