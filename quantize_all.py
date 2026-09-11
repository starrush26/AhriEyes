import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = "models"
targets = [
    ("efficientnet.onnx", "efficientnet_int8.onnx"),
    ("convnext.onnx", "convnext_int8.onnx"),
    ("vit.onnx", "vit_int8.onnx")
]

def quantize_model(input_name, output_name):
    input_path = os.path.join(MODELS_DIR, input_name)
    output_path = os.path.join(MODELS_DIR, output_name)
    temp_clean_path = os.path.join(MODELS_DIR, f"temp_{input_name}")

    if not os.path.exists(input_path):
        print(f"[스킵] {input_path} 파일이 존재하지 않습니다.")
        return

    print(f"\n==========================================")
    print(f"[{input_name}] INT8 양자화 시작...")
    
    # 1. 모델 로드 및 value_info 메타데이터 비우기 (Shape 충돌 방지)
    model = onnx.load(input_path, load_external_data=True)
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()

    # 2. 임시 클린 모델 저장
    onnx.save_model(model, temp_clean_path, save_as_external_data=True)

    # 3. INT8 동적 양자화 수행 (MatMul, Gemm, Conv 레이어 타겟)
    print(f"[{input_name}] 동적 가중치 양자화 연산 중...")
    quantize_dynamic(
        model_input=temp_clean_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Gemm', 'Conv']
    )

    # 4. 임시 파일 삭제
    if os.path.exists(temp_clean_path):
        os.remove(temp_clean_path)
    temp_data = temp_clean_path + ".data"

    if os.path.exists(temp_data):
        os.remove(temp_data)

    print(f"[{output_name}] 압축 완료!")
    
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"-> 최종 파일 크기: {size_mb:.2f} MB")

if __name__ == "__main__":
    for in_m, out_m in targets:
        quantize_model(in_m, out_m)
    print("\n모든 모델 INT8 양자화 완료!")