import cv2
import numpy as np

# 전처리 파라미터 (학습 때와 동일하게!)
SMOOTH_SIZE = 9
COLOR_MIX = 67
EDGE_SHARP = 1

def preprocess_user_method(img_bgr):
    d = SMOOTH_SIZE if SMOOTH_SIZE > 0 else 1
    processed = cv2.bilateralFilter(img_bgr, d, COLOR_MIX, 75)

    if EDGE_SHARP > 0:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + EDGE_SHARP, -1],
                           [-1, -1, -1]], dtype=np.float32)
        kernel = kernel / np.sum(kernel)
        processed = cv2.filter2D(processed, -1, kernel)

    return processed

# # 입력 영상
# in_path  = r"C:\Users\leedg\Documents\lastCV\video.mp4"
# # 출력 영상
# out_path = r"C:\Users\leedg\Documents\lastCV\video_pp.mp4"

in_path  = r"C:\Users\leedg\Documents\lastCV\video2.mp4"
out_path = r"C:\Users\leedg\Documents\lastCV\video2_pp.mp4"



cap = cv2.VideoCapture(in_path)
if not cap.isOpened():
    raise RuntimeError("비디오 열기 실패")

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_pp = preprocess_user_method(frame)
    writer.write(frame_pp)

cap.release()
writer.release()

print("전처리 완료:", out_path)
