import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore")

# --- CONFIG (matches your "Run 3" notebook cells) ---
MODEL_PATH = 'best_model.pth'   # put this file in the same folder as app.py
IMG_SIZE = 224
THRESHOLD = 0.3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
target_layer = model.features[-1]  # same layer your notebook Grad-CAM uses

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# --- BEN GRAHAM PREPROCESSING (must match training exactly) ---
def preprocess_bengrah(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    blur = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # uint8 RGB, 224x224 — good for both the model input and the display/overlay

# --- GRAD-CAM (same hook-based approach as your notebook's GradCAMFixed) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        gradients = self.gradients[0].numpy()     # (C, H, W)
        activations = self.activations[0].numpy() # (C, H, W)
        weights = gradients.mean(axis=(1, 2))      # global average pool

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output

gradcam = GradCAM(model, target_layer)

def make_overlay(processed_img_rgb, cam, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap_color, alpha, processed_img_rgb, 1 - alpha, 0)
    return overlay, heatmap_color

# --- UI ---
st.title("Diabetic Retinopathy Screener")
st.write("EfficientNet-B4 (PyTorch) + Ben Graham Preprocessing + Grad-CAM")

file = st.file_uploader("Upload Retinal Scan", type=["jpg", "jpeg", "png"])

if file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(file)

    with col1:
        st.image(image, caption='Original Upload', use_container_width=True)

    if st.button("Analyze Retina", type="primary"):
        with st.spinner('Applying Ben Graham preprocessing & scanning...'):

            # 1. PREPROCESS
            processed_img = preprocess_bengrah(image)

            with col2:
                st.image(processed_img, caption='Model Input (Processed)', use_container_width=True)

            # 2. PREPARE TENSOR
            input_tensor = transform(processed_img).unsqueeze(0)
            input_tensor.requires_grad_(True)

            # 3. PREDICT + GRAD-CAM IN ONE PASS
            cam, output = gradcam.generate(input_tensor)
            score = torch.sigmoid(output).item()

            # 4. RESULTS
            st.divider()
            if score > THRESHOLD:
                st.error("⚠️ POSITIVE: Diabetic Retinopathy Detected")
                st.progress(int(score * 100))
                st.write(f"**Confidence:** {score*100:.2f}%")
                st.warning("Recommendation: Refer to Ophthalmologist immediately.")
            else:
                st.success("✅ NEGATIVE: Eye looks Healthy")
                st.progress(int((1 - score) * 100))
                st.write(f"**Confidence:** {(1-score)*100:.2f}%")

            # 5. XAI — GRAD-CAM EXPLANATION
            st.divider()
            st.subheader("🔍 Where the model is looking (Grad-CAM)")

            overlay, heatmap_color = make_overlay(processed_img, cam)
            col3, col4 = st.columns(2)
            with col3:
                st.image(heatmap_color, caption='Raw Heatmap', use_container_width=True)
            with col4:
                st.image(overlay, caption='Grad-CAM Overlay', use_container_width=True)

            st.caption(
                "Red/yellow regions show where the model focused most heavily "
                "when producing this prediction."
            )
