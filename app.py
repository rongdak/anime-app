import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="二次元转换器", page_icon="🎨")

MODEL_FILE = "2_4_paprika.onnx"

def process_image(image):
    """
    预处理：强制调整为 512x512，满足静态模型要求
    """
    image = np.array(image.convert('RGB'))
    
    # --- 核心修改：不再计算比例，直接强制 Resize 到 512x512 ---
    # 这样做虽然可能让图片稍微压扁一点，但能保证模型绝对不报错
    image = cv2.resize(image, (512, 512))
    
    image = image.astype(np.float32)
    image = image / 127.5 - 1.0
    
    # HWC -> CHW (通道前置)
    image = image.transpose(2, 0, 1) 
    
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
    fake_img = fake_img.squeeze()
    fake_img = fake_img.transpose(1, 2, 0) # 换回 HWC
    
    fake_img = (fake_img + 1.0) * 127.5
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(fake_img)

# --- 主页面 ---
st.title("🎨 AI 动漫绘图")
st.markdown("### ⚡ 极速版 (512x512)")

uploaded_file = st.file_uploader("请上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    
    # 显示原图
    st.image(original_image, caption="原图", use_column_width=True)
    
    if st.button("⚡ 立即转换", type="primary"):
        with st.spinner("AI 正在绘图..."):
            try:
                anime_image = run_inference(original_image)
                if anime_image:
                    st.image(anime_image, caption="生成结果", use_column_width=True)
                    
                    # 增加下载按钮
                    buf = io.BytesIO()
                    anime_image.save(buf, format="PNG")
                    st.download_button(
                        label="📥 保存图片",
                        data=buf.getvalue(),
                        file_name="anime_result.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"出错: {e}")
# 补充缺失的io库
import io
