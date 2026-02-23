import os
import random
import glob
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

import cv2
from ultralytics import YOLO

# ==============================
# 0) 설정
# ==============================
RAW_DIR = "archive"
RAW_IMG_DIR = os.path.join(RAW_DIR, "images")
RAW_XML_DIR = os.path.join(RAW_DIR, "annotations")

OUT_DIR = "dataset_yolo"
SEED = 42

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

CLASS_NAME = "pothole"
CLASS_ID_MAP = {CLASS_NAME: 0}

SAVE_EXT = ".jpg"   # 전처리 저장은 jpg로 통일

# ==============================
# 전처리 파라미터
# ==============================
SMOOTH_SIZE = 9      # d
COLOR_MIX = 67       # sigmaColor
EDGE_SHARP = 1       # Sharpness

# ==============================
# 1) 유틸
# ==============================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def imread_imdecode(path: str):
    """너 코드 방식: 한글 경로 호환 읽기"""
    try:
        n = np.fromfile(path, np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def imwrite_imencode(path: str, img_bgr):
    """너 코드 방식: imencode + tofile 저장"""
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".jpg"
        path = path + ext

    result, encoded_img = cv2.imencode(ext, img_bgr)
    if not result:
        return False
    with open(path, mode="w+b") as f:
        encoded_img.tofile(f)
    return True

def preprocess_user_method(img_bgr):
    """
    - bilateralFilter(d, sigmaColor=COLOR_MIX, sigmaSpace=75)
    - sharpen kernel (9 + EDGE_SHARP), 정규화 포함
    - Resize 없음
    """
    d = SMOOTH_SIZE if SMOOTH_SIZE > 0 else 1
    processed = cv2.bilateralFilter(img_bgr, d, COLOR_MIX, 75)

    if EDGE_SHARP > 0:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + EDGE_SHARP, -1],
                           [-1, -1, -1]], dtype=np.float32)
        kernel = kernel / np.sum(kernel)
        processed = cv2.filter2D(processed, -1, kernel)

    return processed

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    w = int(size.findtext("width"))
    h = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))
        objects.append((name, xmin, ymin, xmax, ymax))

    return filename, w, h, objects

def voc_to_yolo_line(class_id, xmin, ymin, xmax, ymax, img_w, img_h):
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))

    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    return f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"

# ==============================
# 2) 데이터 목록 정리 (이미지-XML 매칭)
# ==============================
def collect_pairs():
    xml_paths = sorted(glob.glob(os.path.join(RAW_XML_DIR, "*.xml")))
    pairs = []

    for xp in xml_paths:
        filename, _, _, _ = parse_voc_xml(xp)

        img_path = os.path.join(RAW_IMG_DIR, filename)
        if not os.path.exists(img_path):
            stem = Path(filename).stem
            cand = glob.glob(os.path.join(RAW_IMG_DIR, stem + ".*"))
            if len(cand) == 0:
                print(f"[WARN] 이미지 없음: xml={xp} filename={filename}")
                continue
            img_path = cand[0]

        pairs.append((img_path, xp))

    return pairs

# ==============================
# 3) split
# ==============================
def split_pairs(pairs):
    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"Total: {n} | train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs

# ==============================
# 4) 전처리 + YOLO 라벨 생성 + 저장
# ==============================
def build_yolo_dataset(train_pairs, val_pairs, test_pairs):
    for split_name, split_pairs_ in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        img_out_dir = os.path.join(OUT_DIR, "images", split_name)
        lbl_out_dir = os.path.join(OUT_DIR, "labels", split_name)
        ensure_dir(img_out_dir)
        ensure_dir(lbl_out_dir)

        for img_path, xml_path in split_pairs_:
            img = imread_imdecode(img_path)
            if img is None:
                print(f"[WARN] 이미지 로드 실패: {img_path}")
                continue

            img_pp = preprocess_user_method(img)

            # XML 파싱
            filename, w, h, objects = parse_voc_xml(xml_path)

            # 이미지 저장명 통일
            stem = Path(filename).stem
            out_img_path = os.path.join(img_out_dir, stem + SAVE_EXT)

            if not imwrite_imencode(out_img_path, img_pp):
                print(f"[WARN] 이미지 저장 실패: {out_img_path}")
                continue

            # 라벨(txt) 생성 (좌표는 원본 기준 그대로 사용 → Resize 없으니 OK)
            yolo_lines = []
            for (name, xmin, ymin, xmax, ymax) in objects:
                if name not in CLASS_ID_MAP:
                    continue
                class_id = CLASS_ID_MAP[name]
                yolo_lines.append(voc_to_yolo_line(class_id, xmin, ymin, xmax, ymax, w, h))

            out_lbl_path = os.path.join(lbl_out_dir, stem + ".txt")
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

    # data.yaml 생성
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {Path(OUT_DIR).resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n\n"
            f"names:\n"
            f"  0: {CLASS_NAME}\n"
        )
    print(f"[OK] data.yaml 생성: {yaml_path}")
    return yaml_path

# ==============================
# 5) YOLOv8n 학습
# ==============================
# ==============================
# 5) 최적화된 학습 설정 (Tuning)
# ==============================
# ==============================
# 5) 최종 성능 돌파 설정 (Hybrid)
# ==============================
def train_yolov8(data_yaml_path):
    # [중요] n모델 대신 s모델 사용
    # 처음엔 yolov8s.pt가 없어서 다운로드하느라 시간이 좀 걸릴 수 있습니다.
    model = YOLO("yolov8s.pt")

    print("===== TRAIN START (Model S - High Performance) =====")
    
    results = model.train(
        data=data_yaml_path,
        epochs=150,      
        patience=20,     
        
        # [핵심 타협] 사양 때문에 해상도를 낮추고 배치를 극도로 줄입니다.
        imgsz=512,       # 640은 i3에서 s모델 돌리기에 너무 무거움
        batch=4,         # 8GB 램 방어용 (4나 2로 설정)
        
        # [최적화]
        optimizer='AdamW',
        lr0=0.001,
        
        # [데이터 증강] 포트홀에 효과적인 증강
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,       # [추가] 두 이미지를 겹쳐서 학습 (일반화 성능 UP)
        
        name="pothole_yolov8s",
        project="runs/detect",
        device="cpu",
        plots=True
    )
    
    # ... (이하 결과 출력 코드는 동일) ...
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    return best_pt, save_dir

# ==============================
# main
# ==============================
def main():
    pairs = collect_pairs()
    if len(pairs) == 0:
        raise RuntimeError("매칭된 (이미지, xml) 쌍이 0개입니다. 폴더 구조/파일명을 확인하세요.")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    yaml_path = build_yolo_dataset(train_pairs, val_pairs, test_pairs)
    best_pt, run_dir = train_yolov8(yaml_path)

if __name__ == "__main__":
    main()
