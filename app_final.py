import streamlit as st
from ultralytics import YOLO
import cv2
import os
import time
import json
import glob
import random
import yt_dlp
import shutil
from datetime import datetime
import pandas as pd

# ==========================================
# [설정] 챔피언 모델 경로 (본인 경로로 수정 필수!)
# ==========================================
MODEL_PATH = r"runs\detect\runs\detect\pothole_yolov8s\weights\best.pt"

# 데이터 저장소
DB_DIR = "pothole_db"
if not os.path.exists(DB_DIR): os.makedirs(DB_DIR)

st.set_page_config(page_title="자동 이어보기 관제 시스템", layout="wide", page_icon="🚦")

# ==========================================
# 💾 상태 관리 (Session State)
# ==========================================
if 'detections' not in st.session_state: st.session_state.detections = []
if 'work_orders' not in st.session_state: st.session_state.work_orders = []

# [핵심 1] 팝업 제어 및 영상 상태 변수
if 'show_popup' not in st.session_state: st.session_state.show_popup = False
if 'popup_data' not in st.session_state: st.session_state.popup_data = None
if 'video_playing' not in st.session_state: st.session_state.video_playing = False
if 'current_frame' not in st.session_state: st.session_state.current_frame = 0 # 프레임 위치 기억

# ==========================================
# 📍 함수 정의 (이어보기 로직 포함)
# ==========================================
def resume_video():
    """팝업을 닫고 영상을 1초 뒤로 넘겨서 재생"""
    st.session_state.show_popup = False
    st.session_state.popup_data = None
    st.session_state.video_playing = True
    
    # [핵심 2] 방금 잡은 포트홀을 또 잡지 않게 30프레임(약 1초) 건너뛰기
    st.session_state.current_frame += 30 
    
    # 화면을 새로고침해서 영상 루프를 다시 실행시킴
    st.rerun()

def move_to_work_order(item):
    item['status'] = "작업지시완료"
    item['order_time'] = datetime.now().strftime("%H:%M:%S")
    st.session_state.work_orders.append(item)
    st.toast(f"✅ 작업 지시 전송 완료! 영상이 이어집니다.")
    resume_video() # 작업 후 자동 재개

def delete_detection():
    st.toast("🗑️ 데이터 삭제 완료! 영상이 이어집니다.")
    resume_video() # 삭제 후 자동 재개

# ==========================================
# 📍 유틸리티 & 모델
# ==========================================
@st.cache_resource
def load_model(): return YOLO(MODEL_PATH)

def get_mock_address():
    districts = ["유성구 어은동", "강남구 역삼동", "서초구 서초동", "분당구 정자동", "해운대구 우동"]
    return f"대전광역시 {random.choice(districts)} {random.randint(1, 999)}번지"

def get_mock_gps():
    return 36.3634 + random.uniform(-0.01, 0.01), 127.3559 + random.uniform(-0.01, 0.01)

def download_youtube_video(url):
    ydl_opts = {'format': 'best[ext=mp4]', 'outtmpl': 'temp_stream.mp4', 'overwrites': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'temp_stream.mp4'

# ==========================================
# 🖥️ 메인 UI
# ==========================================
st.title("🚦 포트홀 도로 관제 시스템")
st.markdown("---")

# 🌟 팝업창 (Modal) - 버튼 누르면 resume_video() 호출
if st.session_state.show_popup and st.session_state.popup_data:
    item = st.session_state.popup_data
    with st.expander("🚨 긴급! 도로 파손 감지 (영상 일시정지 중)", expanded=True):
        c_info, c_act = st.columns([3, 1])
        with c_info:
            st.info(f"📍 {item['address']} | 🕒 {item['time']}")
            t1, t2 = st.tabs(["탐지 화면", "원본 화면"])
            t1.image(item['img_detect'], use_column_width=True)
            t2.image(item['img_orig'], use_column_width=True)
        with c_act:
            st.write("### 조치 선택")
            if st.button("✅ 작업 전송", type="primary", use_container_width=True):
                move_to_work_order(item)
            if st.button("🗑️ 오탐지 삭제", type="secondary", use_container_width=True):
                delete_detection()
            if st.button("▶️ 그냥 계속 보기", use_container_width=True):
                resume_video()
    # 팝업이 떠있을 땐 아래 영상 코드가 실행되지 않도록 여기서 멈춤
    st.stop()

# 메인 화면 레이아웃
col_video, col_list = st.columns([1.5, 1])

with col_video:
    st.subheader("📺 영상 관제")
    input_source = st.radio("소스 선택", ["🔗 유튜브", "📂 파일"], horizontal=True)
    
    video_path = "temp_stream.mp4" # 기본 경로
    
    if input_source == "🔗 유튜브":
        url = st.text_input("YouTube URL")
        # URL이 있고 파일이 없으면 다운로드
        if url and st.button("영상 준비 (다운로드)"):
            with st.spinner("다운로드 중..."):
                download_youtube_video(url)
                st.session_state.current_frame = 0 # 새 영상이니 초기화
                st.success("준비 완료!")
    else:
        file = st.file_uploader("파일 업로드", type=['mp4', 'avi'])
        if file:
            with open("temp_stream.mp4", "wb") as f: f.write(file.read())
            st.session_state.current_frame = 0 # 새 영상이니 초기화
            video_path = "temp_stream.mp4"

    # 제어 버튼
    c1, c2 = st.columns(2)
    if c1.button("🚀 관제 시작 / 재개", type="primary", use_container_width=True):
        st.session_state.video_playing = True
        st.rerun()
        
    if c2.button("⏹️ 초기화 (처음부터)", type="secondary", use_container_width=True):
        st.session_state.video_playing = False
        st.session_state.current_frame = 0 # 프레임 초기화
        st.rerun()
        
    video_placeholder = st.empty()

with col_list:
    st.subheader("📋 처리 현황")
    tab1, tab2 = st.tabs([f"접수 ({len(st.session_state.detections)})", f"작업지시 ({len(st.session_state.work_orders)})"])
    with tab1:
        for item in reversed(st.session_state.detections):
            st.caption(f"🔴 {item['time']} - {item['id']}")
    with tab2:
        for item in reversed(st.session_state.work_orders):
            st.success(f"👷 {item['time']} - {item['id']}")

# ==========================================
# 🚀 영상 재생 엔진 (이어보기 핵심 로직)
# ==========================================
if st.session_state.video_playing and not st.session_state.show_popup:
    
    # 파일이 실제로 있는지 확인
    if not os.path.exists("temp_stream.mp4"):
        st.warning("분석할 영상이 없습니다. 유튜브 링크를 넣거나 파일을 업로드하세요.")
        st.session_state.video_playing = False
    else:
        cap = cv2.VideoCapture("temp_stream.mp4")
        model = load_model()

        # [핵심 3] 저장된 프레임 위치로 점프!
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if st.session_state.current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
        else:
            st.success("영상이 종료되었습니다.")
            st.session_state.video_playing = False

        # 재생 루프
        while cap.isOpened() and st.session_state.video_playing:
            ret, frame = cap.read()
            if not ret:
                st.session_state.video_playing = False
                st.session_state.current_frame = 0
                st.rerun()
                break
            
            # 현재 위치 실시간 업데이트
            st.session_state.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 추론
            results = model.predict(frame, conf=0.15, verbose=False)
            res_plotted = results[0].plot()
            video_placeholder.image(res_plotted, channels="BGR", use_column_width=True)
            
            # 감지됨?
            if len(results[0].boxes) > 0:
                # 팝업 데이터 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_data = {
                    "id": f"RAD_{st.session_state.current_frame}",
                    "address": get_mock_address(),
                    "gps": f"{get_mock_gps()}",
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "img_detect": res_plotted,
                    "img_orig": frame
                }
                
                # 상태 업데이트 (멈춤 & 팝업 오픈)
                st.session_state.popup_data = new_data
                st.session_state.show_popup = True
                st.session_state.detections.append(new_data)
                
                # 자원 해제 후 리런 (화면 갱신을 위해 필수)
                cap.release()
                st.rerun()
                break

        cap.release()