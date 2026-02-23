import cv2
from ultralytics import YOLO

# ==========================================
# [설정] 백업한 최종 모델 경로를 여기에 넣으세요
# ==========================================
MODEL_PATH = r"runs\detect\runs\detect\pothole_yolov8s\weights\best.pt"

# 테스트하고 싶은 동영상이나 이미지 경로 (0을 넣으면 웹캠이 켜집니다)
# SOURCE = "https://www.youtube.com/watch?v=Jr7k5wEpAc8"  # 또는 "test_image.jpg" 또는 0
#SOURCE = "https://www.youtube.com/watch?v=SyIQirLZB7A"




# SOURCE = r"C:\Users\leedg\Documents\lastCV\video.mp4"


SOURCE = r"C:\Users\leedg\Documents\lastCV\video2_pp.mp4"

def run_inference():
    # 1. 모델 로드
    print(f"🏆 챔피언 모델 로드 중: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception:
        print("❌ 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요!")
        return

    # 2. 추론 실행 (우리가 찾은 최적값 conf=0.15 적용)
    print("🚀 포트홀 탐지 시작... (종료하려면 화면 클릭 후 'q' 키 누르세요)")
    
    # predict() 함수로 영상/이미지 실행
    # conf=0.15 : 우리가 찾은 mAP 0.80 달성 기준값
    # save=True : 결과 영상을 파일로 저장
    # show=True : 화면에 실시간으로 보여줌
    model.predict(
        source=SOURCE, 
        conf=0.15,      # [핵심] 이 값을 써야 mAP 0.80 성능이 나옵니다!
        save=True,
        show=True,
        line_width=2    # 박스 두께
    )

    print("\n✅ 탐지 완료! 결과가 'runs/detect/predict...' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    run_inference()