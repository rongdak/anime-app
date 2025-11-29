import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os
import io

st.set_page_config(page_title="二次元转换器", page_icon="🎨")

MODEL_FILE = "2_4_paprika.onnx"

def resize_crop_center(image, target_size=512):
    """
    核心改进：中心裁剪模式
    不拉伸图片，而是截取中间的正方形区域，保证人物不变形。
    """
    h, w = image.shape[:2]
    
    # 1. 计算裁剪区域
    short_edge = min(h, w)
    start_h = (h - short_edge) // 2
    start_w = (w - short_edge) // 2
    
    # 2. 进行裁剪 (得到正方形)
    cropped_img = image[start_h:start_h+short_edge, start_w:start_w+short_edge]
    
    # 3. 缩放到目标大小 (512x512)
    resized_img = cv2.resize(cropped_img, (target_size, target_size))
    return resized_img

def process_image(image):
    image = np.array(image.convert('RGB'))
    
    # 使用新的裁剪函数
    image = resize_crop_center(image)
    
    image = image.astype(np.float32)
    image = image / 127.5 - 1.0
    image = image.transpose(2, 0, 1) # HWC -> CHW
    image = np.expand_dims(image, axis=0)
    return image

def run_inference(image_pil):
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ 找不到模型文件 {MODEL_FILE}，请确保已上传到GitHub。")
        st.stop()

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(MODEL_FILE, sess_options)
    except Exception as e:
        st.error(f"❌ 模型加载出错，可能是文件没传对。详细错误: {e}")
        st.stop()

    x_name = session.get_inputs()[0].name
    y_name = session.get_outputs()[0].name
    
    img_input = process_image(image_pil)
    
    # 推理
    fake_img = session.run([y_name], {x_name: img_input})[0]
    
    # 后处理
    fake_img = fake_img.squeeze()
    fake_img = fake_img.transpose(1, 2, 0)
    fake_img = (fake_img + 1.0) * 127.5
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(fake_img)

# --- 主页面 ---
st.title("🎨 AI 动漫绘图 (宫崎骏版)")
st.info("💡虽然背景会变少，但人脸会清晰很多！建议上传人脸较大的照片。")

uploaded_file = st.file_uploader("请上传照片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    # 提示用户将进行裁剪
    st.image(original_image, caption="原图 (将截取中心正方形区域)", use_column_width=True)
    
    if st.button("⚡ 立即转换", type="primary"):
        with st.spinner("AI 正在精细绘制中..."):
            try:
                anime_image = run_inference(original_image)
                if anime_image:
                    st.image(anime_image, caption="宫崎骏风格效果", use_column_width=True)
                    
                    # 下载按钮
                    buf = io.BytesIO()
                    anime_image.save(buf, format="PNG")
                    st.download_button(
                        label="📥 保存高清大图",
                        data=buf.getvalue(),
                        file_name="anime_hayao.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"出错啦: {e}")
