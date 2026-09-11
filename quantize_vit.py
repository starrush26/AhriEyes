import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model_path = os.path.join("models", "vit.onnx")
temp_clean_path = os.path.join("models", "vit_clean.onnx")
output_model_path = os.path.join("models", "vit_int8.onnx")

print("1. 충돌 유발하는 기존 Shape 메타데이터 초기화 중...")
model = onnx.load(input_model_path, load_external_data=True)

# 형상 검사 충돌을 일으키는 중간 value_info 메타데이터 전부 삭제
while len(model.graph.value_info) > 0:
    model.graph.value_info.pop()

# 임시 정리 모델 저장
onnx.save_model(model, temp_clean_path, save_as_external_data=True)

print("2. ViT INT8 동적 양자화 수행 중...")
quantize_dynamic(
    model_input=temp_clean_path,
    model_output=output_model_path,
    weight_type=QuantType.QInt8, # 8비트 양자화
    op_types_to_quantize=['MatMul', 'Gemm']
)

# 임시 파일 정리
if os.path.exists(temp_clean_path): # 임시 정리 모델 삭제
    os.remove(temp_clean_path)
data_clean = temp_clean_path + ".data" # 임시 정리 데이터 파일 삭제

if os.path.exists(data_clean): # 임시 정리 데이터 파일 삭제
    os.remove(data_clean)

print(f"\n양자화 완료! 생성 파일: {output_model_path}")

if os.path.exists(output_model_path): # 양자화 후 모델 파일 존재 여부 확인
    print(f"압축 전: {os.path.getsize(input_model_path) / (1024*1024):.2f} MB")
    print(f"압축 후: {os.path.getsize(output_model_path) / (1024*1024):.2f} MB")