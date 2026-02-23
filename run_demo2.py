import cv2
from ultralytics import YOLO

MODEL_PATH = r"runs\detect\runs\detect\pothole_yolov8s\weights\best.pt"
SOURCE = r"C:\Users\leedg\Documents\lastCV\video2_pp.mp4"

def run_inference():

    print(f"🏆 챔피언 모델 로드 중: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print("❌ 영상 열기 실패")
        return

    print("🚀 포트홀 탐지 시작 (아래 60%만)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # 🔥 아래 60%만 사용
        roi_start = int(h * 0.4)
        roi = frame[roi_start:h, 0:w]

        # ROI만 모델에 입력
        results = model.predict(
            source=roi,
            conf=0.15,
            verbose=False
        )

        # 결과 표시
        annotated = results[0].plot()

        cv2.imshow("ROI Detection (60%)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
