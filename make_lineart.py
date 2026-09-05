import cv2
import numpy as np

# 1. 원본 이미지 불러오기
src = cv2.imread('static/spirit_blossom_ahri.png', cv2.IMREAD_UNCHANGED)
if src is None:
    print("파일을 찾을 수 없습니다.")
    exit()

if len(src.shape) == 3 and src.shape[2] == 4:
    bgr = src[:, :, :3]
else:
    bgr = src

# 2. 얼굴 이목구비 대비 향상 및 필터링
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=40, sigmaSpace=40)

# 3. 엣지 검출 (이전과 동일한 최적 좌표)
edges = cv2.Canny(filtered, 35, 110)

# 4. 순수 흰색 배경 영역 엣지 제외
white_bg = (bgr[:, :, 0] > 240) & (bgr[:, :, 1] > 240) & (bgr[:, :, 2] > 240)
kernel = np.ones((3, 3), np.uint8)
white_bg_eroded = cv2.erode(white_bg.astype(np.uint8), kernel, iterations=1)
edges[white_bg_eroded == 1] = 0

# 5. [핵심] 원본 색상을 선에 입히고 네온처럼 은은하게 밝기/채도 부스팅
# HSV 공간으로 변환해 어두운 선도 은은하게 빛나도록 명도와 채도 보정
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255) # 채도 은은하게 강화
hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4 + 40, 0, 255) # 선이 어두운 배경에 묻히지 않게 명도 업
boosted_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# 6. 최종 RGBA 합성
h, w = edges.shape
out = np.zeros((h, w, 4), dtype=np.uint8)

# 엣지 픽셀 위치에 부스팅된 원본 색상과 은은한 투명도(220) 주입
mask = edges > 0
out[mask, :3] = boosted_bgr[mask]
out[mask, 3] = 225

cv2.imwrite('static/spirit_blossom_ahri.webp', out)
print("✅ 원본 컬러 기반 은은한 네온 라인아트 생성 완료!")