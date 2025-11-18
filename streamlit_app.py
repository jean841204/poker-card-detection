import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path
import tempfile
import numpy as np

# 設定頁面配置
st.set_page_config(
    page_title="USPS 數字辨識器",
    page_icon="🔢",
    layout="wide"
)

# 標題
st.title("🔢 USPS 數字辨識系統")
st.markdown("---")

# 側邊欄 - 模型資訊
with st.sidebar:
    st.header("📋 系統資訊")
    st.info("使用 YOLOv8 進行數字分類")

    # 模型設定
    st.header("⚙️ 推論設定")
    confidence = st.slider("信心度閾值", 0.0, 1.0, 0.3, 0.05)
    img_size = st.number_input("圖片大小", min_value=16, max_value=640, value=32, step=16)

# 載入模型
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "weight" / "best.pt"
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return None

model = load_model()

if model is None:
    st.error("⚠️ 無法載入模型，請確認 weight/best.pt 檔案存在")
    st.stop()

# 顯示模型類別
if hasattr(model, 'names'):
    with st.sidebar:
        st.header("🏷️ 可辨識類別")
        st.write(model.names)

# 推論函數
def predict_image(image, model, conf, imgsz):
    """對圖片進行推論"""
    try:
        # 使用臨時檔案儲存上傳的圖片
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        # 進行推論
        results = model.predict(
            source=tmp_path,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )

        # 清理臨時檔案
        os.unlink(tmp_path)

        return results
    except Exception as e:
        st.error(f"推論錯誤: {e}")
        return None

# 顯示推論結果
def display_results(results, image):
    """顯示推論結果"""
    if results is None or len(results) == 0:
        st.warning("沒有檢測到任何結果")
        return

    result = results[0]

    # 顯示圖片
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 原始圖片")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 推論結果")

        if result.probs is not None:
            # 分類結果
            probs = result.probs
            top5_indices = probs.top5
            top5_conf = probs.top5conf.cpu().numpy()

            st.write("**Top 5 預測結果:**")

            for idx, conf in zip(top5_indices, top5_conf):
                class_name = model.names[int(idx)]
                st.metric(
                    label=f"類別: {class_name}",
                    value=f"{conf*100:.2f}%"
                )

            # 最高信心度預測
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            predicted_class = model.names[int(top1_idx)]

            st.success(f"### 🎯 預測結果: **{predicted_class}**")
            st.info(f"信心度: **{top1_conf*100:.2f}%**")
        else:
            st.warning("無法獲取分類結果")

# 主要內容區域 - 使用 radio 選擇模式
if 'page_mode' not in st.session_state:
    st.session_state['page_mode'] = "📤 上傳圖片"

# 當選擇範例圖片時，自動切換到範例圖片模式
if 'selected_example_image' in st.session_state:
    st.session_state['page_mode'] = "📁 選擇範例圖片"

page_mode = st.radio(
    "選擇輸入方式",
    ["📤 上傳圖片", "📁 選擇範例圖片"],
    horizontal=True,
    key='page_mode'
)

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
