import streamlit as st
import os
import requests
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="二次元转换器", page_icon="🎨")

# --- 关键修改：改了文件名，强制系统重新下载 ---
MODEL_URL = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/2_4_paprika.onnx"
MODEL_FILE = "anime_model_v2.onnx" # 改名了，这样系统就不会用之前那个坏文件了

def download_model():
    """下载模型，增加防报错机制"""
    if not os.path.exists(MODEL_FILE):
        st.info("🚀 正在下载 AI 模型 (约8MB)，请耐心等待 1-2 分钟...")
        try:
            # 增加 headers 伪装成浏览器，防止被拦截
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(MODEL_URL, headers=headers, stream=True)
            
            with open(MODEL_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            # 检查下载是否成功（文件是否太小）
            if os.path.getsize(MODEL_FILE) < 1000000: # 如果小于1MB肯定不对
                os.remove(MODEL_FILE)
                st.error("❌ 模型下载失败（文件过小），请刷新页面重试。")
                st.stop()
                
            st.success("✅ 模型下载成功！")
        except Exception as e:
            st.error(f"❌ 下载出错: {e}")
            st.stop()

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
    download_model() # 确保模型存在
    
    try:
        session = ort.InferenceSession(MODEL_FILE)
    except Exception as e:
        # 如果加载失败，自动删除坏文件，提示用户刷新
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        st.error(f"模型文件损坏，已自动删除。请刷新页面重试！错误信息: {e}")
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

st.title("🎨 照片转动漫神器")
st.write("上传照片，一键变身宫崎骏画风！")

uploaded_file = st.file_uploader("请选择一张图片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="原图", use_column_width=True)
    
    if st.button("⚡ 开始转换", type="primary"):
        with st.spinner("正在施展魔法..."):
            anime_image = run_inference(original_image)
            if anime_image:
                st.image(anime_image, caption="动漫效果", use_column_width=True)import streamlit as st
import os
import requests
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

# --- 1. 页面配置 (必须放在第一行) ---
st.set_page_config(page_title="二次元转换器", page_icon="🎨")

# --- 2. 核心设置 ---
# 模型下载地址 (使用Paprika风格，效果较好)
MODEL_URL = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/2_4_paprika.onnx"
MODEL_FILE = "model.onnx"

def download_model_if_needed():
    """检查模型是否存在，不存在则自动下载"""
    if not os.path.exists(MODEL_FILE):
        progress_text = st.empty()
        progress_text.info("🚀 首次运行，正在下载AI模型 (约8MB)...请稍候")
        try:
            r = requests.get(MODEL_URL)
            with open(MODEL_FILE, 'wb') as f:
                f.write(r.content)
            progress_text.success("✅ 模型下载完成！")
        except Exception as e:
            progress_text.error(f"❌ 下载失败: {e}")
            st.stop()

def process_image(image, size=512):
    """图片预处理"""
    image = np.array(image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    
    # 缩放图片，避免内存溢出
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
    """执行AI转换"""
    download_model_if_needed()
    
    # 加载模型
    session = ort.InferenceSession(MODEL_FILE)
    x_name = session.get_inputs()[0].name
    y_name = session.get_outputs()[0].name
    
    img_input = process_image(image_pil)
    if img_input is None: return None
    
    # 推理
    fake_img = session.run([y_name], {x_name: img_input})[0]
    
    # 后处理
    fake_img = fake_img.squeeze()
    fake_img = (fake_img + 1.0) * 127.5
    fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(fake_img)

# --- 3. 界面设计 ---
st.title("🎨 照片转动漫神器")
st.markdown("不用去日本，一键生成宫崎骏画风！")

uploaded_file = st.file_uploader("点击上传一张照片 (人像/风景)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="原图", use_column_width=True)
    
    if st.button("⚡ 开始转换", type="primary"):
        with st.spinner("AI 正在疯狂绘画中..."):
            try:
                anime_image = run_inference(original_image)
                
                st.success("转换成功！")
                st.image(anime_image, caption="动漫效果", use_column_width=True)
                
                # 下载按钮
                buf = io.BytesIO()
                anime_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 保存图片",
                    data=byte_im,
                    file_name="anime_result.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"出错啦: {e}")

