import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2  # You need OpenCV for the Ben Graham processing

# 1. HIDE WARNINGS
import warnings
warnings.filterwarnings("ignore")

# 2. LOAD MODEL
@st.cache_resource
def load_model():
    # Ensure this matches your downloaded model name exactly
    return tf.keras.models.load_model('V2_Final_Best_Model.keras')

model = load_model()

# --- HELPER FUNCTIONS (The "Ben Graham" Code) ---
# We must use the exact same logic as your training script

def crop_image_from_gray(img, tol=7):
    # This detects the eye circle and cuts off the black corners
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): return img 
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_input(image_pil):
    # 1. Convert PIL Image (RGB) to NumPy Array (RGB)
    img_array = np.array(image_pil)
    
    # 2. Ben Graham Preprocessing (Circle Crop + Blur)
    # Note: Training used BGR->RGB conversion. Since PIL is already RGB,
    # we treat it carefully to match training conditions.
    
    # Crop the black borders
    img_array = crop_image_from_gray(img_array)
    
    # Resize to 300x300 (Matches V2 model input)
    img_array = cv2.resize(img_array, (300, 300))
    
    # Apply Gaussian Blur (The "Ben Graham" signature)
    img_array = cv2.addWeighted(img_array, 4, cv2.GaussianBlur(img_array, (0,0), 10), -4, 128)
    
    return img_array

# --- UI LAYOUT ---

st.title("Diabetic Retinopathy Screener (Phase 1)")
st.write("Using EfficientNetB3 + Ben Graham Preprocessing")

file = st.file_uploader("Upload Retinal Scan", type=["jpg", "png", "jpeg"])

if file is not None:
    col1, col2 = st.columns(2)
    
    # Open Image
    image = Image.open(file)
    
    with col1:
        st.image(image, caption='Original Upload', use_container_width=True)

    if st.button("Analyze Retina", type="primary"):
        with st.spinner('Applying Ben Graham Preprocessing & Scanning...'):
            
            # 1. PREPROCESS (The Fix)
            # We transform the image exactly how the model learned it
            processed_img = preprocess_input(image)
            
            # Show the "Computer Vision" view (Great for XAI demo!)
            with col2:
                st.image(processed_img, caption='Model Input (Processed)', use_container_width=True)

            # 2. PREPARE FOR MODEL
            # Normalize (0-1) because we used rescale=1./255 in training
            final_input = processed_img / 255.0
            
            # Add Batch Dimension (1, 300, 300, 3)
            final_input = np.expand_dims(final_input, axis=0)

            THRESHOLD = 0.35

            # 3. PREDICT
            prediction = model.predict(final_input)
            score = prediction[0][0]

            # 4. RESULTS
            st.divider()
            
            if score > THRESHOLD:
                # SICK
                st.error(f"⚠️ POSITIVE: Diabetic Retinopathy Detected")
                # Show confidence bar
                st.progress(int(score*100))
                st.write(f"**Confidence:** {score*100:.2f}%")
                st.warning("Recommendation: Refer to Ophthalmologist immediately.")
            else:
                # HEALTHY
                st.success(f"✅ NEGATIVE: Eye looks Healthy")
                st.progress(int((1-score)*100))
                st.write(f"**Confidence:** {(1-score)*100:.2f}%")