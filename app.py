import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os
import io

st.set_page_config(page_title="二次元风格转换", page_icon="🎨", layout="wide")

# --- 1. 这里对应你截图里的两个文件名 ---
STYLES = {
    "宫崎骏风 (Hayao) - 线条清晰": "hayao.onnx",
    "新海诚风 (Shinkai) - 风景唯美": "shinkai.onnx"
}

def resize_crop_center(image, target_size=512):
    """中心裁剪，保证不变形"""
    h, w = image.shape[:2]
    short_edge = min(h, w)
    start_h = (h - short_edge) // 2
    start_w = (w - short_edge) // 2
    cropped_img = image[start_h:start_h+short_edge, start_w:start_w+short_edge]
    resized_img = cv2.resize(cropped_img, (target_size, target_size))
    return resized_img

def process_image(image, style_name):
    image = np.array(image.convert('RGB'))
    image = resize_crop_center(image)
    image = image.astype(np.float32)
    
    # 统一归一化
    image = image / 127.5 - 1.0
    
    # --- 关键逻辑：根据文件名判断处理方式 ---
    if "shinkai" in style_name:
        # 新海诚 (Shinkai) 保持 HWC，不动
        pass
    else:
        # 宫崎骏 (Hayao) 需要变为 CHW
        image = image.transpose(2, 0, 1)
        
    image = np.expand_dims(image, axis=0)
    return image

def post_process(output, style_name):
    output = output.squeeze()
    
    # --- 关键逻辑：还原 ---
    if "shinkai" in style_name:
        pass
    else:
        # 宫崎骏 需要变回 HWC
        output = output.transpose(1, 2, 0)
        
    # 反归一化
    output = (output + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return Image.fromarray(output)

def run_inference(image_pil, model_filename):
    if not os.path.exists(model_filename):
        st.error(f"❌ 找不到模型文件: {model_filename}")
        st.warning("请检查 GitHub 仓库里是否上传了该文件，名字必须完全一样！")
        return None

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(model_filename, sess_options)
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None

    x_name = session.get_inputs()[0].name
    y_name = session.get_outputs()[0].name
    
    img_input = process_image(image_pil, model_filename)
    
    # 推理
    fake_img = session.run([y_name], {x_name: img_input})[0]
    
    # 后处理
    result_img = post_process(fake_img, model_filename)
    return result_img

# --- 页面 UI ---
st.title("🎨 AI 动漫双风格生成器")
st.markdown("### 上传照片，在 宫崎骏 和 新海诚 之间切换！")

# 侧边栏
with st.sidebar:
    st.header("🎨 风格选择")
    selected_style = st.radio("请选择画风:", list(STYLES.keys()))
    current_model = STYLES[selected_style]
    st.info(f"当前加载: {current_model}")

# 主区域
uploaded_file = st.file_uploader("请上传图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原图")
        st.image(original_image, use_column_width=True)

    with col2:
        st.subheader("生成结果")
        if st.button("✨ 立即生成", type="primary"):
            with st.spinner(f"正在绘制 {selected_style.split(' - ')[0]} 风格..."):
                anime_image = run_inference(original_image, current_model)
                
                if anime_image:
                    st.image(anime_image, use_column_width=True)
                    
                    buf = io.BytesIO()
                    anime_image.save(buf, format="PNG")
                    st.download_button(
                        label="📥 保存高清大图",
                        data=buf.getvalue(),
                        file_name=f"anime_{current_model.split('.')[0]}.png",
                        mime="image/png"
                    )
