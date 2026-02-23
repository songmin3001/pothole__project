# check_result.py (학습 없이 결과만 확인하는 코드)
from ultralytics import YOLO
from pathlib import Path

# 사진에 보이는 'best.pt'의 경로를 여기에 복사해서 넣으세요
# 예: "runs/detect/runs/detect/weights/best.pt" (경로 중첩 주의)
BEST_PT_PATH = r"runs\detect\runs\detect\pothole_yolov8s\weights\best.pt"

def check_metrics():
    model_path = Path(BEST_PT_PATH)
    
    if not model_path.exists():
        print("경로를 확인해주세요. 파일을 찾을 수 없습니다.")
        return

    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # 평가 실행 (data.yaml 경로가 필요할 수 있음, 보통 모델 안에 정보가 저장됨)
    # 만약 에러나면 data='dataset_yolo/data.yaml' 인자를 추가하세요.
    metrics = model.val(split='test', device='cpu')
    
    print("\n" + "="*30)
    print(f" mAP@50     : {metrics.box.map50:.4f}") 
    print(f" Precision  : {metrics.box.mp:.4f}")
    print(f" Recall     : {metrics.box.mr:.4f}")
    print("="*30)

if __name__ == "__main__":
    check_metrics()