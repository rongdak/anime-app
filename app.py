import streamlit as st
import os
import requests
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="二次元转换器", page_icon="🎨")

# 强制使用新文件名，避免读取到旧的损坏文件
MODEL_URL = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/2_4_paprika.onnx"
MODEL_FILE = "anime_model_v2.onnx"

def download_model():
    # 检查模型是否存在
    if not os.path.exists(MODEL_FILE):
        st.info("🚀 正在下载 AI 模型 (约8MB)，请耐心等待...")
        try:
            # 伪装浏览器头信息
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(MODEL_URL, headers=headers, stream=True)
            
            with open(MODEL_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            # 校验文件大小，防止下载空文件
            if os.path.getsize(MODEL_FILE) < 1000000:
                os.remove(MODEL_FILE)
                st.error("❌ 下载失败：文件过小，请刷新页面重试")
                st.stop()
                
            st.success("✅ 模型下载成功！")
        except Exception as e:
            st.error(f"❌ 下载出错: {e}")
            st.stop()

def process_image(image, size=512):
    # 图片预处理
    image = np.array(image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = new_h - (new_h % 32)
    new_w = new_w - (new_w % 32)
    
    if new_h == 0 or new_w == 0: return None
    
    image = cv2.resize(image, (new_w, new_h))
    image = image.astype(np.float32)
    image = image / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)
    return image

def run_inference(image_pil):
    download_model()
    
    try:
        session = ort.InferenceSession(MODEL_FILE)
    except Exception as e:
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        st.error(f"模型加载失败，已自动清理坏文件。请刷新页面重试！\n错误: {e}")
        st.stop()

    x_name = session.get_inputs()[0].name
    y_name = session.get_outputs()[0].name
    
    img_input = process_image(image_pil)
    if img_input is None: return None
    
    fake_img = session.run([y_name], {x_name: img_input})[0]
    
    fake_img = fake_img.squeeze()
    fake_img = (fake_img + 1.0) * 127.5
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(fake_img)

# --- 主页面 ---
st.title("🎨 照片转动漫神器")
st.write("上传照片，一键生成二次元形象！")

uploaded_file = st.file_uploader("请上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="原图", use_column_width=True)
    
    if st.button("⚡ 开始转换", type="primary"):
        with st.spinner("正在生成中..."):
            anime_image = run_inference(original_image)
            if anime_image:
                st.image(anime_image, caption="动漫效果", use_column_width=True)
