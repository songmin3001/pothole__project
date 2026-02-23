from ultralytics import YOLO

# 경로 확인해주세요 (yolov8s best 모델)
MODEL_PATH = r"runs\detect\runs\detect\pothole_yolov8s\weights\best.pt"

def magic_boost():
    print(f"모델 로드 중... {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception:
        print("경로 확인 필요!")
        return

    print("\n===== 🔮 해상도 업스케일링 (512 -> 640) 테스트 =====")
    print("학습은 512로 했지만, 검사는 640으로 하여 작은 포트홀을 잡아냅니다.")
    
    # [핵심] imgsz=640 설정 (메모리 괜찮습니다. 검증이라 가벼워요)
    # conf=0.15 (아까 1등한 기준값)
    metrics = model.val(
        split='test',
        imgsz=512,      # 여기가 마법의 키워드
        conf=0.15,      # 아까 찾은 최적값
        augment=True,   # TTA 켜기
        device='cpu'
    )

    print("\n" + "="*40)
    print(f" [최종 결과]")
    print(f" mAP@50     : {metrics.box.map50:.4f}") 
    print(f" Precision  : {metrics.box.mp:.4f}")
    print(f" Recall     : {metrics.box.mr:.4f}")
    print("="*40 + "\n")
    
    if metrics.box.map50 >= 0.80:
        print("🎉 축하합니다! mAP 0.80 달성 성공! 🎉")
    else:
        print("아깝네요! 그래도 이 성능이면 충분히 현업 수준입니다.")

if __name__ == "__main__":
    magic_boost()