import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os
import io

st.set_page_config(page_title="新海诚风格转换", page_icon="🌤️")

MODEL_FILE = "Shinkai_53.onnx"

def resize_crop_center(image, target_size=512):
    """中心裁剪，保证不变形"""
    h, w = image.shape[:2]
    short_edge = min(h, w)
    start_h = (h - short_edge) // 2
    start_w = (w - short_edge) // 2
    cropped_img = image[start_h:start_h+short_edge, start_w:start_w+short_edge]
    resized_img = cv2.resize(cropped_img, (target_size, target_size))
    return resized_img

def process_image(image):
    image = np.array(image.convert('RGB'))
    image = resize_crop_center(image)
    
    image = image.astype(np.float32)
    
    # --- 修复点 1：归一化 ---
    image = image / 127.5 - 1.0
    
    # --- 修复点 2：不要 transpose (保持 HWC 格式) ---
    # 删掉了 image = image.transpose(2, 0, 1)
    
    # 增加 batch 维度: (1, 512, 512, 3)
    image = np.expand_dims(image, axis=0)
    return image

def run_inference(image_pil):
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ 找不到模型文件 {MODEL_FILE}")
        st.stop()

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(MODEL_FILE, sess_options)
    except Exception as e:
        st.error(f"❌ 模型加载出错: {e}")
        st.stop()

    x_name = session.get_inputs()[0].name
    y_name = session.get_outputs()[0].name
    
    img_input = process_image(image_pil)
    
    # 推理
    fake_img = session.run([y_name], {x_name: img_input})[0]
    
    # 后处理
    fake_img = fake_img.squeeze() # 去掉 batch 维度
    
    # --- 修复点 3：输出也是 HWC，不需要转回 ---
    # 删掉了 fake_img = fake_img.transpose(1, 2, 0)
    
    fake_img = (fake_img + 1.0) * 127.5
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(fake_img)

# --- 主页面 ---
st.title("🌤️ AI 动漫绘图 (新海诚版)")
st.info("💡 唯美光影风格，中心裁剪模式。")

uploaded_file = st.file_uploader("请上传照片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="原图", use_column_width=True)
    
    if st.button("⚡ 立即转换", type="primary"):
        with st.spinner("正在绘制..."):
            try:
                anime_image = run_inference(original_image)
                if anime_image:
                    st.image(anime_image, caption="新海诚效果", use_column_width=True)
                    
                    buf = io.BytesIO()
                    anime_image.save(buf, format="PNG")
                    st.download_button(
                        label="📥 保存图片",
                        data=buf.getvalue(),
                        file_name="shinkai_style.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"出错: {e}")
