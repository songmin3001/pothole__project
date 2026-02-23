# pothole__project
## pothole Detection using YOLOv8n with Tiling & Padding Optimization

### Project Background
> 도로 포트홀은 차량 파손, 타이어 손상, 교통사고를 유발하는 주요 원인 중 하나이며, 
지자체의 유지보수 비용 증가와 직결되는 인프라 관리 문제이다. 
특히 대도시의 경우 도로 길이가 방대하여 인력 기반 점검 방식은 시간과 비용 측면에서 한계가 존재한다.

> 최근 컴퓨터 비전 기술을 활용한 자동 탐지 시스템이 주목받고 있으나,
실제 환경에서는 다음과 같은 어려움이 존재한다.

1. 포트홀은 작은 객체에 해당하며 탐지가 어렵다.
2. 조명, 노면 질감, 카메라 흔들림 등 환경 변화가 크다.
3. 실시간 관제 시스템 적용을 위해 연산 효율성이 요구된다.
4. CPU 환경에서도 구동 가능해야 한다.

> 따라서 본 프로젝트는 단순히 YOLO 모델을 적용하는 것이 아니라,
전처리 전략, 학습 파라미터 튜닝, confidence threshold 최적화,
그리고 ROI 기반 추론 구조 설계를 통해 실제 도로 관제 시스템에 적용 가능한 최적화된 포트홀 탐지 파이프라인 을 구축하는 것을 목표로 한다.

### How to run
본 프로젝트는 다음 3단계로 실행된다.

1. Dataset 생성 및 전처리
2. YOLOv8 모델 학습
3. ROI 기반 실시간 추론 실행

#### 1.Environment Setup
1. Python 3.9 이상 권장
필수 라이브러리 설치 :
```
pip install ultralytics opencv-python numpy pandas
```

```
pip install ultralytics opencv-python streamlit yt-dlp numpy pandas
```

#### 2. Dataset 생성 + 전처리
+ 원본 데이터 구조
archive/
 ├── images/
 └── annotations/  (Pascal VOC XML)

+ 실행
```
python train.py
```

+ 실행 process
- VOC → YOLO 포맷 변환
- 전처리 적용
- train / val / test random split
- data.yaml 생성

+ 생성 결과
dataset_yolo/
 ├── images/
 │    ├── train/
 │    ├── val/
 │    └── test/
 ├── labels/
 └── data.yaml

#### 3. Model Training
```
 runs/detect/pothole_yolov8s/weights/best.pt
```

#### 4. 성능 확인
```
 python check_result.py
```

 + 출력 예시
===== TEST SET PERFORMANCE =====
mAP@50     : 0.7443
Precision  : 0.7901
Recall     : 0.6584

#### 5. Confidence Threshold 최적화
```
python find_sweet_spot.py
```

#### 6. 최종 ROI 기반 추론 실행
```
python inference_roi.py
```

#### 7. 최종 실행 
```
streamlit run app_final.py
```

#### 8. 순서 요약
1. train.py 실행
2. check_result.py 실행
3. find_sweet_spot.py 실행
4. inference_roi.py 실행


### 문제 해결 방법
아래는 프로젝트 실행 중 자주 발생하는 문제와 해결 방법입니다.


#### 1. best.pt 파일을 찾을 수 없습니다. (FileNotFoundError)
+ 증상
- `경로를 확인해주세요. 파일을 찾을 수 없습니다.`
- `Missing best.pt`

+ 원인 
- `MODEL_PATH`가 잘못됨 (특히 `runs/detect`가 중복되는 경우가 많음)
- 학습이 완료되지 않아 best.pt가 생성되지 않음

+ 해결
1. 실제 파일 위치를 먼저 확인합니다.

예시(정상 경로):
- `runs/detect/pothole_yolov8s/weights/best.pt`

2. `app_final.py`, `inference_roi.py` 등에서 아래처럼 수정합니다.

Mac/Linux:
```python
MODEL_PATH = "runs/detect/pothole_yolov8s/weights/best.pt"
```

Windows:
```python
MODEL_PATH = r"runs\detect\pothole_yolov8s\weights\best.pt"
```

#### 2. streamlit 실행이 안 됩니다. (command not found / ModuleNotFound)

+ 증상
- streamlit: command not found
- ModuleNotFoundError: streamlit

+ 원인
- streamlit이 설치되지 않았거나, 현재 파이썬 환경과 설치 환경이 다름

+ 해결
```
python -m pip install streamlit
python -m streamlit run app_final.py
```

#### 3. 유튜브 영상 다운로드가 실패합니다. (yt-dlp 관련)

+ 증상
- 다운로드 버튼을 눌러도 영상이 생성되지 않음
- yt-dlp 에러 발생

+ 원인
- yt-dlp 미설치 또는 일부 유튜브 영상은 제한/차단

+ 해결
1. 설치 확인
```
python -m pip install yt-dlp
```

2. 유튜브 대신 파일 업로드 모드로 테스트 권장
3. Streamlit에서 📂 파일 선택 후 mp4 업로드

#### 4. 영상이 분석되지 않습니다. (temp_stream.mp4 없음)

+ 증상
- 분석할 영상이 없습니다. 유튜브 링크를 넣거나 파일을 업로드하세요.

+ 원인
- temp_stream.mp4 파일이 아직 생성되지 않음
- 다운로드/업로드 전에 "관제 시작"을 눌렀음

+ 해결
1. 유튜브: 영상 준비(다운로드) 먼저 클릭
2. 파일: mp4 업로드 후 자동 저장 확인
3. 그 다음 관제 시작/재개 버튼 클릭

#### 5. 팝업이 계속 뜨고 영상이 안 이어집니다. (중복 감지)

+ 증상
- 포트홀이 한 번 감지된 이후 같은 지점에서 계속 감지됨

+ 원인
- 감지 직후 같은 프레임 근처에서 다시 탐지되는 상황
- 이어보기 로직이 제대로 적용되지 않음

+ 해결
- resume_video()에서 감지 직후 프레임을 건너뛰도록 설정되어 있어야 함
- st.session_state.current_frame += 30  # 약 1초 스킵
- 필요 시 30 → 60으로 늘려서 중복 감지를 더 줄일 수 있음

### Dataset
- Source: Kaggle dataset - https://www.kaggle.com/datasets/andrewmvd/pothole-detection/data
- Classes: 1 - pothole
- Total Images: 665
- Train / Val / Test split: 80 / 10 / 10
- Annotation format: YOLOv8n
