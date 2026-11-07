import cv2
import numpy as np
import mediapipe as mp

# MediaPipe 얼굴 감지 솔루션 초기화
mp_face_detection = mp.solutions.face_detection

def has_human_face(image_bytes: bytes) -> bool:
    """
    이미지 바이트에서 1인 이상의 사람 얼굴이 명확히 존재하는지 검증
    - min_detection_confidence를 0.75로 하여 동물/사물 오탐지 원천 차단
    """
    try:
        # 바이트를 OpenCV BGR 이미지로 디코딩
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        # BGR -> RGB 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 0.5 -> 0.75로 기준치를 올려서 동물 눈코입 오탐지 컷트
        with mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.75
        ) as face_detector:
            results = face_detector.process(img_rgb)

            # 감지된 객체가 없으면 즉시 차단
            if not results.detections:
                return False

            # 감지된 얼굴의 바운딩 박스 신뢰도 재검증
            for detection in results.detections:
                score = detection.score[0] if detection.score else 0.0
                if score >= 0.75:
                    return True

            return False

    except Exception as e:
        # 에러 발생 시 로그 출력
        print(f"[FaceFilter Error] {e}")
        return False