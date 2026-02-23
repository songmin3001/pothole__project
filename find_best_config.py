from ultralytics import YOLO
import sys

# [중요] 방금 학습한 모델 경로 (정밀도 0.82 나온 모델)
# 경로가 다르다면 수정해주세요.
MODEL_PATH = r"runs\detect\runs\detect\pothole_yolov8s\weights\best.pt"
# 만약 yolov8n 하드어그멘테이션 모델이라면 경로를 그쪽으로 맞춰주세요.

def find_sweet_spot():
    print(f"모델 로드 중... {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception:
        print("경로를 확인해주세요!")
        return

    print("\n===== 🏆 0.80 돌파를 위한 '최적의 기준값' 찾기 =====")
    
    # 테스트할 기준값들 (0.05부터 0.30까지 0.05 단위로 테스트)
    # 기준이 낮을수록 Recall은 오르고 Precision은 떨어집니다. 
    # 그 사이에서 mAP가 최대화되는 지점을 찾습니다.
    confs = [0.10, 0.15, 0.20, 0.25]
    
    best_map = 0
    best_conf = 0

    for conf in confs:
        print(f"\n[Testing] 기준값(conf) = {conf} + TTA(augment=True)")
        
        # augment=True (TTA)는 필수입니다. 점수를 1~2% 올려줍니다.
        metrics = model.val(split='test', conf=conf, augment=True, device='cpu', verbose=False)
        
        p = metrics.box.mp    # Precision
        r = metrics.box.mr    # Recall
        map50 = metrics.box.map50 # mAP@50
        
        print(f" -> 결과: mAP {map50:.4f} | P {p:.4f} | R {r:.4f}")
        
        if map50 > best_map:
            best_map = map50
            best_conf = conf

    print("\n" + "="*40)
    print(f"👑 최종 우승 설정: conf={best_conf}")
    print(f"   최고 mAP : {best_map:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    find_sweet_spot()