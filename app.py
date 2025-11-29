import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="二次元转换器", page_icon="🎨")

# 你的模型文件名 (确保GitHub上也是这个名字)
MODEL_FILE = "2_4_paprika.onnx"

def process_image(image, size=512):
    """
    预处理：
    1. 调整大小
    2. 归一化
    3. 关键修改：HWC -> CHW (把通道移到前面)
    """
    image = np.array(image.convert('RGB'))
    # 注意：这个模型需要 RGB 格式，不要转 BGR
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    
    h, w = image.shape[:2]
    
    # 调整大小逻辑
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = new_h - (new_h % 32)
    new_w = new_w - (new_w % 32)
    
    if new_h == 0 or new_w == 0: return None
    
    image = cv2.resize(image, (new_w, new_h))
    image = image.astype(np.float32)
    image = image / 127.5 - 1.0
    
    # --- 修复核心：维度置换 ---
    # 原图是 (High, Width, Channel)，模型要 (Channel, High, Width)
    image = image.transpose(2, 0, 1) 
    
    image = np.expand_dims(image, axis=0)
    return image

def run_inference(image_pil):
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ 找不到模型文件 {MODEL_FILE}，请检查GitHub上传是否成功。")
        st.stop()

    try:
        # 禁用一些优化以提高兼容性
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(MODEL_FILE, sess_options)
    except Exception as e:
        st.error(f"❌ 模型加载出错: {e}")
        st.stop()

    x_name = session.get_inputs()[0].name
    y_name = session.get_outputs()[0].name
    
    img_input = process_image(image_pil)
    if img_input is None: return None
    
    # 推理
    fake_img = session.run([y_name], {x_name: img_input})[0]
    
    # 后处理：把维度换回来
    fake_img = fake_img.squeeze() # 去掉 batch 维度
    # (Channel, High, Width) -> (High, Width, Channel)
    fake_img = fake_img.transpose(1, 2, 0) 
    
    fake_img = (fake_img + 1.0) * 127.5
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(fake_img)

# --- 主页面 ---
st.title("🎨 AI 动漫绘图")
st.write("已加载 FacePaint 油画风格模型")

uploaded_file = st.file_uploader("请上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="原图", use_column_width=True)
    
    if st.button("⚡ 立即转换", type="primary"):
        with st.spinner("AI 正在绘图，请稍候..."):
            try:
                anime_image = run_inference(original_image)
                if anime_image:
                    st.image(anime_image, caption="生成结果", use_column_width=True)
            except Exception as e:
                st.error(f"运行出错: {e}")
