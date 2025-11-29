import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="二次元转换器", page_icon="🎨")

# 直接使用你刚上传的文件名
MODEL_FILE = "2_4_paprika.onnx"

def process_image(image, size=512):
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
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ 找不到模型文件！请确认你已经把 {MODEL_FILE} 上传到了 GitHub 仓库里。")
        st.stop()

    try:
        session = ort.InferenceSession(MODEL_FILE)
    except Exception as e:
        st.error(f"❌ 模型加载出错: {e}")
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
        with st.spinner("AI 正在绘制中..."):
            anime_image = run_inference(original_image)
            if anime_image:
                st.image(anime_image, caption="动漫效果", use_column_width=True)
