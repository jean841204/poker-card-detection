import streamlit as st
from PIL import Image
import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import cv2
import torch

# 添加 YOLOv7 路徑到系統路徑
# 優先使用專案內的 yolov7，其次使用本地路徑
YOLOV7_PATH = Path(__file__).parent / "yolov7"
if not YOLOV7_PATH.exists():
    YOLOV7_PATH = Path("/Users/jessica/Desktop/NCHU/研究方法論/yolov7")

if YOLOV7_PATH.exists():
    sys.path.insert(0, str(YOLOV7_PATH))

# 設定頁面配置
st.set_page_config(
    page_title="撲克牌花色辨識器",
    page_icon="🃏",
    layout="wide"
)

# 標題
st.title("🃏 撲克牌花色辨識系統")
st.markdown("---")

# 側邊欄 - 模型資訊
with st.sidebar:
    st.header("📋 系統資訊")
    st.info("使用 YOLOv7 進行撲克牌花色分類")

    # 模型設定
    st.header("⚙️ 推論設定")
    confidence = st.slider("信心度閾值", 0.0, 1.0, 0.25, 0.05)
    img_size = st.number_input("圖片大小", min_value=320, max_value=1280, value=640, step=32)

# 載入模型
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "weight" / "best.pt"

    # 方法 1: 嘗試使用本地 YOLOv7（最可靠）
    if YOLOV7_PATH.exists():
        try:
            # 導入本地 YOLOv7 的 models
            from models.experimental import attempt_load
            from utils.torch_utils import select_device

            device = select_device('cpu')  # 使用 CPU
            model = attempt_load(str(model_path), map_location=device)
            model.conf = 0.25
            model.iou = 0.45

            # 包裝模型以支援 YOLOv7 的推論介面
            class YOLOv7Wrapper:
                def __init__(self, model, device):
                    self.model = model
                    self.device = device
                    self.conf = 0.25
                    self.iou = 0.45
                    self.names = model.names if hasattr(model, 'names') else model.module.names

                def __call__(self, img, size=640):
                    from utils.general import non_max_suppression, scale_coords
                    from utils.datasets import letterbox
                    import torch

                    # 預處理圖片
                    img0 = img.copy()
                    img = letterbox(img, size, stride=32)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(self.device)
                    img = img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # 推論
                    with torch.no_grad():
                        pred = self.model(img)[0]

                    # NMS
                    pred = non_max_suppression(pred, self.conf, self.iou)

                    # 處理檢測結果
                    class Results:
                        def __init__(self, pred, img0, img, names):
                            self.pred = pred
                            self.img0 = img0
                            self.img = img
                            self.names = names

                        def pandas(self):
                            import pandas as pd
                            class XYXYContainer:
                                def __init__(self, pred, names):
                                    self.data = []
                                    if pred[0] is not None and len(pred[0]):
                                        for *xyxy, conf, cls in pred[0].cpu().numpy():
                                            self.data.append({
                                                'xmin': xyxy[0],
                                                'ymin': xyxy[1],
                                                'xmax': xyxy[2],
                                                'ymax': xyxy[3],
                                                'confidence': conf,
                                                'class': int(cls),
                                                'name': names[int(cls)]
                                            })
                                    self.xyxy = [pd.DataFrame(self.data)]

                                def __getitem__(self, idx):
                                    return self.xyxy[idx]

                                def __len__(self):
                                    return len(self.xyxy)

                            return XYXYContainer(self.pred, self.names)

                        def render(self):
                            from utils.plots import plot_one_box
                            img = self.img0.copy()
                            if self.pred[0] is not None and len(self.pred[0]):
                                for *xyxy, conf, cls in self.pred[0].cpu().numpy():
                                    label = f'{self.names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, img, label=label, line_thickness=2)
                            return [img]

                    # 調整檢測框到原圖大小
                    for det in pred:
                        if det is not None and len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    return Results(pred, img0, img, self.names)

            wrapped_model = YOLOv7Wrapper(model, device)
            st.sidebar.success("✅ 使用本地 YOLOv7 載入模型")
            return wrapped_model

        except Exception as e:
            st.sidebar.warning(f"本地載入失敗，嘗試使用 torch.hub: {e}")

    # 方法 2: 使用 torch.hub（備用）
    try:
        # 設定安全的全域變數（PyTorch 2.6+ 需要）
        import numpy
        try:
            torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
        except:
            pass

        model = torch.hub.load('WongKinYiu/yolov7', 'custom',
                               path_or_model=str(model_path),
                               force_reload=False,
                               trust_repo=True,
                               _verbose=False)
        model.conf = 0.25
        model.iou = 0.45
        st.sidebar.success("✅ 使用 torch.hub 載入模型")
        return model
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return None

model = load_model()

if model is None:
    st.error("⚠️ 無法載入模型，請確認 weight/best.pt 檔案存在")
    st.stop()

# 顯示模型類別
try:
    with st.sidebar:
        st.header("🏷️ 可辨識類別")
        if hasattr(model, 'names'):
            st.write(model.names)
        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
            st.write(model.module.names)
except:
    pass

# 推論函數
def predict_image(image, model, conf, imgsz):
    """對圖片進行推論"""
    try:
        # 設定模型參數
        model.conf = conf

        # 將 PIL Image 轉換為 numpy array
        img_np = np.array(image)

        # 如果是 RGBA，轉換為 RGB
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        # 進行推論
        results = model(img_np, size=imgsz)

        return results
    except Exception as e:
        st.error(f"推論錯誤: {e}")
        return None

# 影片推論函數 - 生成處理後的影片
def predict_video(video_path, model, conf, imgsz, process_every_frame=True):
    """對影片進行推論並生成帶有檢測框的新影片"""
    try:
        cap = cv2.VideoCapture(video_path)

        # 取得影片資訊
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 建立輸出影片
        # 使用 H.264 編碼以確保瀏覽器可以播放
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        # 嘗試使用 H.264 編碼（最廣泛支援）
        # 如果失敗則退回到 mp4v
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264
            cv2.VideoWriter_fourcc(*'H264'),  # H.264 alternative
            cv2.VideoWriter_fourcc(*'X264'),  # H.264 alternative
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 fallback
        ]

        out = None
        for fourcc in fourcc_options:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                break

        if not out or not out.isOpened():
            st.error("無法建立影片編碼器")
            return None, None, None, None

        frame_count = 0
        all_detections = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 將 OpenCV BGR 格式轉換為 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 進行推論
            model.conf = conf
            results = model(frame_rgb, size=imgsz)

            # 收集檢測結果用於統計
            try:
                detections = results.pandas().xyxy[0]
                for _, row in detections.iterrows():
                    all_detections.append(row['name'])
            except:
                pass

            # 取得帶有檢測框的圖片
            rendered_frame = np.squeeze(results.render())

            # 轉回 BGR 用於 OpenCV 寫入
            frame_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            # 更新進度
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"處理進度: {frame_count}/{total_frames} 幀")

        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()

        return output_path, all_detections, fps, total_frames
    except Exception as e:
        st.error(f"影片處理錯誤: {e}")
        return None, None, None, None

# 顯示推論結果
def display_results(results, image):
    """顯示推論結果（YOLOv7 檢測格式）"""
    if results is None:
        st.warning("沒有檢測到任何結果")
        return

    try:
        # 獲取檢測結果
        detections = results.pandas().xyxy[0]
    except Exception as e:
        st.error(f"無法解析檢測結果: {e}")
        return

    # 顯示圖片
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 原始圖片")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 檢測結果圖")
        # 顯示帶有檢測框的圖片
        rendered_img = np.squeeze(results.render())
        st.image(rendered_img, use_container_width=True)

    # 顯示檢測詳情
    st.subheader("📊 檢測詳情")

    if len(detections) == 0:
        st.info("未檢測到任何物體")
    else:
        # 統計各類別數量
        class_counts = detections['name'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**檢測到的物體:**")
            for idx, row in detections.iterrows():
                st.metric(
                    label=f"物體 {idx + 1}: {row['name']}",
                    value=f"{row['confidence']*100:.2f}%"
                )

        with col2:
            st.write("**類別統計:**")
            for class_name, count in class_counts.items():
                st.write(f"- **{class_name}**: {count} 個")

        # 顯示最高信心度的檢測
        top_detection = detections.loc[detections['confidence'].idxmax()]
        st.success(f"### 🎯 最高信心度: **{top_detection['name']}** ({top_detection['confidence']*100:.2f}%)")

# 主要內容區域 - 使用 radio 選擇模式
if 'page_mode' not in st.session_state:
    st.session_state['page_mode'] = "📤 上傳圖片"

page_mode = st.radio(
    "選擇輸入方式",
    ["📤 上傳圖片", "🎬 上傳影片", "📁 選擇範例圖片"],
    horizontal=True,
    key='page_mode'
)

# 當切換模式時，清除範例圖片選擇
if page_mode == "📤 上傳圖片" and 'selected_example_image' in st.session_state:
    del st.session_state['selected_example_image']

st.markdown("---")

# 上傳圖片模式
if page_mode == "📤 上傳圖片":
    st.header("上傳你的圖片")
    uploaded_file = st.file_uploader(
        "選擇一張圖片...",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="支援 PNG, JPG, JPEG, BMP 格式"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 開始辨識", use_container_width=True, type="primary"):
                with st.spinner("辨識中..."):
                    results = predict_image(image, model, confidence, img_size)
                    if results:
                        st.markdown("---")
                        display_results(results, image)

# 上傳影片模式
elif page_mode == "🎬 上傳影片":
    st.header("上傳你的影片")

    uploaded_video = st.file_uploader(
        "選擇一個影片檔案...",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支援 MP4, AVI, MOV, MKV 格式"
    )

    if uploaded_video is not None:
        # 儲存上傳的影片到臨時檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        st.markdown("---")

        # 顯示原始影片預覽
        with st.expander("📹 原始影片預覽", expanded=False):
            st.video(video_path)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 開始辨識影片", use_container_width=True, type="primary"):
                with st.spinner("處理影片中，請稍候..."):
                    output_video_path, all_detections, fps, total_frames = predict_video(
                        video_path, model, confidence, img_size
                    )

                    if output_video_path:
                        st.markdown("---")
                        st.success(f"### ✅ 影片處理完成！")
                        st.info(f"影片資訊: 總幀數 {total_frames}，FPS {fps}")

                        # 嘗試使用 ffmpeg 重新編碼以確保瀏覽器相容性
                        try:
                            import subprocess

                            # 使用 ffmpeg 重新編碼為 H.264
                            re_encoded_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                            # 檢查是否有 ffmpeg
                            result = subprocess.run(
                                ['ffmpeg', '-version'],
                                capture_output=True,
                                timeout=5
                            )

                            if result.returncode == 0:
                                # 使用 ffmpeg 重新編碼
                                subprocess.run([
                                    'ffmpeg', '-y', '-i', output_video_path,
                                    '-c:v', 'libx264',
                                    '-preset', 'fast',
                                    '-crf', '22',
                                    '-c:a', 'copy',
                                    re_encoded_path
                                ], capture_output=True, check=True)

                                # 替換為重新編碼的影片
                                os.unlink(output_video_path)
                                output_video_path = re_encoded_path
                                st.sidebar.info("✅ 使用 ffmpeg 重新編碼")
                            else:
                                st.sidebar.warning("⚠️ ffmpeg 不可用，使用原始編碼")
                        except:
                            # 如果 ffmpeg 失敗，使用原始影片
                            st.sidebar.warning("⚠️ ffmpeg 重新編碼失敗，使用原始編碼")
                            pass

                        # 讀取影片檔案
                        with open(output_video_path, 'rb') as video_file:
                            video_bytes = video_file.read()

                        # 顯示處理後的影片
                        st.subheader("🎬 檢測結果影片")
                        st.video(video_bytes)

                        # 提供下載按鈕
                        st.download_button(
                            label="⬇️ 下載處理後的影片",
                            data=video_bytes,
                            file_name="detected_video.mp4",
                            mime="video/mp4"
                        )

                        # 統計分析
                        if all_detections:
                            st.markdown("---")
                            st.subheader("📈 統計分析")

                            from collections import Counter
                            detection_counts = Counter(all_detections)
                            total_objects = len(all_detections)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**檢測結果統計:**")
                                for class_name, count in detection_counts.most_common():
                                    st.write(f"- {class_name}: {count} 次 ({count/total_objects*100:.1f}%)")

                            with col2:
                                st.write("**最常檢測到的花色:**")
                                most_common = detection_counts.most_common(1)[0]
                                st.metric("花色", most_common[0], f"{most_common[1]} 次")
                                st.metric("總檢測數", total_objects)
                                st.metric("平均每幀檢測數", f"{total_objects/total_frames:.2f}")

                        # 清理臨時檔案
                        try:
                            os.unlink(output_video_path)
                        except:
                            pass

                # 清理輸入影片臨時檔案
                try:
                    os.unlink(video_path)
                except:
                    pass

# 選擇範例圖片模式
elif page_mode == "📁 選擇範例圖片":
    st.header("從資料集中選擇範例圖片")

    # 獲取 data 資料夾中的所有圖片
    data_path = Path(__file__).parent / "data"

    if data_path.exists():
        image_files = sorted(list(data_path.glob("*.png")) + list(data_path.glob("*.jpg")))

        if len(image_files) > 0:
            # 檢查是否已選擇圖片
            if 'selected_example_image' not in st.session_state:
                # 顯示圖片縮圖供選擇
                st.write(f"共有 {len(image_files)} 張範例圖片，點擊「選擇」即可自動辨識")

                # 使用網格佈局顯示圖片
                cols = st.columns(5)

                for idx, img_path in enumerate(image_files):
                    col = cols[idx % 5]
                    with col:
                        img = Image.open(img_path)
                        st.image(img, caption=img_path.name, use_container_width=True)
                        if st.button(f"選擇", key=f"select_{idx}"):
                            # 選擇圖片並自動進行推論
                            st.session_state['selected_example_image'] = img_path
                            st.rerun()
            else:
                # 已選擇圖片，顯示結果
                selected_path = st.session_state['selected_example_image']

                # 顯示縮圖網格（可收合）
                with st.expander("📂 瀏覽其他範例圖片", expanded=False):
                    st.write(f"共有 {len(image_files)} 張範例圖片")
                    cols = st.columns(5)

                    for idx, img_path in enumerate(image_files):
                        col = cols[idx % 5]
                        with col:
                            img = Image.open(img_path)
                            st.image(img, caption=img_path.name, use_container_width=True)
                            if st.button(f"選擇", key=f"select_exp_{idx}"):
                                st.session_state['selected_example_image'] = img_path
                                st.rerun()

                # 顯示辨識結果標題
                st.success(f"### ✅ 已選擇並辨識: {selected_path.name}")

                # 重新選擇按鈕
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 重新選擇其他圖片", use_container_width=True):
                        del st.session_state['selected_example_image']
                        st.rerun()

                st.markdown("---")

                # 載入圖片並進行推論
                image = Image.open(selected_path)

                with st.spinner("辨識中..."):
                    results = predict_image(image, model, confidence, img_size)
                    if results:
                        display_results(results, image)
        else:
            st.warning("data 資料夾中沒有找到圖片")
    else:
        st.error("找不到 data 資料夾")

# 頁尾
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>💡 提示: 調整側邊欄的參數以改變推論設定</p>
    </div>
    """,
    unsafe_allow_html=True
)
