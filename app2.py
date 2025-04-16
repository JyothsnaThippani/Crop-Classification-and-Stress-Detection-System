import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tifffile as tiff
import cv2

# Load U-Net model for .npy
@st.cache_resource
def load_unet_model():
    return load_model(r"C:\Users\gowth\Downloads\segnet_model.h5")

# Load SegNet model for .tif
@st.cache_resource
def load_segnet_model():
    model_path = r"C:\Users\gowth\Downloads\crop_segnet_model_1.h5"
    model = tf.keras.models.load_model(model_path, compile=False)

    def iou_metric(y_true, y_pred, smooth=1e-6):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    def iou_loss(y_true, y_pred):
        return 1 - iou_metric(y_true, y_pred)

    model.compile(optimizer='adam', loss=iou_loss, metrics=[iou_metric])
    return model

unet_model = load_unet_model()
segnet_model = load_segnet_model()

# Labels for U-Net mask
LABELS = {
    0: ("🔴", "Wheat"),
    1: ("🟢", "Cotton"),
    2: ("🔵", "Tomatoes"),
    3: ("🟡", "Rice"),
    4: ("🟣", "Maize"),
    5: ("🟤", "Barley"),
    6: ("🟠", "Sunflower"),
    7: ("🟦", "Sugarcane"),
    8: ("⚪", "Soybean")
}

# Preprocess and predict U-Net mask
def preprocess_npy(file):
    data = np.load(file)
    return data.astype(np.float32)

def predict_unet_mask(image):
    image = np.expand_dims(image, axis=0)
    pred_mask = unet_model.predict(image)[0]
    return np.argmax(pred_mask, axis=-1)

def visualize_unet_results(image, mask):
    st.subheader("🖼️ Crop Classification Result")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image[..., 0], cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, (color, _) in LABELS.items():
        if label in mask:
            rgb_color = plt.get_cmap('tab10')(label)[:3]
            mask_rgb[mask == label] = [int(c * 255) for c in rgb_color]

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Predicted Crop Mask")
    axes[1].axis("off")

    st.pyplot(fig)

# NDVI Calculation for SegNet
def calculate_ndvi(image, red_band=3, nir_band=7):
    red = image[:, :, red_band].astype(float)
    nir = image[:, :, nir_band].astype(float)
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

# Crop Recommendation Logic
def get_recommendation(mean_ndvi):
    if mean_ndvi >= 0.6:
        return "🌿 Healthy condition. Keep up with regular management."
    elif 0.4 <= mean_ndvi < 0.6:
        return "💧 Moderate stress detected. Increase irrigation and observe plant health."
    elif 0.2 <= mean_ndvi < 0.4:
        return "⚠️ High stress detected. Apply nutrients and adjust water management."
    else:
        return "🚨 Severe stress detected! Immediate action recommended for pests or replanting."

def visualize_ndvi(image, model):
    st.subheader("🌾 NDVI Analysis and Stress Detection")
    ndvi = calculate_ndvi(image)
    ndvi_resized = cv2.resize(ndvi, (64, 64), interpolation=cv2.INTER_AREA)

    pred_mask = model.predict(ndvi_resized[np.newaxis, ..., np.newaxis])[0].squeeze()

    mean_ndvi = np.mean(ndvi_resized)
    min_ndvi = np.min(ndvi_resized)
    max_ndvi = np.max(ndvi_resized)
    std_ndvi = np.std(ndvi_resized)
    var_ndvi = np.var(ndvi_resized)

    if var_ndvi < 0.015:
        stress_type = "🌀 Abiotic Stress (uniform pattern)"
    else:
        stress_type = "🦠 Biotic Stress (patchy pattern)"

    recommendation = get_recommendation(mean_ndvi)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(ndvi, cmap='RdYlGn')
    axes[0].set_title("NDVI Image")

    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title("Predicted Stress Mask")

    st.pyplot(fig)

    st.markdown("### 📊 NDVI Statistics")
    st.write(f"Minimum NDVI: {min_ndvi:.4f}")
    st.write(f"Maximum NDVI: {max_ndvi:.4f}")
    st.write(f"Mean NDVI: {mean_ndvi:.4f}")
    st.write(f"Standard Deviation: {std_ndvi:.4f}")
    st.write(f"Variance: {var_ndvi:.6f}")

    st.markdown("### 🔍 Stress Type Assessment")
    st.write(stress_type)
    st.markdown("### 📢 Recommendation")
    st.write(recommendation)

st.title("🌱 Crop Classification and Stress Detection System")

st.header("📂 1. Upload Crop Classification Data (.npy)")
npy_file = st.file_uploader("Upload a .npy file for classification", type=["npy"], key="npy")

if npy_file is not None:
    image_data = preprocess_npy(npy_file)
    st.success("✅ Image loaded successfully!")
    predicted_mask = predict_unet_mask(image_data)
    visualize_unet_results(image_data, predicted_mask)

    st.subheader("🎨 Class Color Legend")
    for label, (color, name) in LABELS.items():
        st.markdown(f"{color} {name}")

st.header("📂 2. Upload NDVI Image (.tif)")
tif_file = st.file_uploader("Upload a TIFF image for NDVI analysis", type=["tif", "tiff"], key="tif")

if tif_file is not None:
    try:
        image = tiff.imread(tif_file)
        st.success("✅ TIFF image loaded successfully!")
        visualize_ndvi(image, segnet_model)
    except Exception as e:
        st.error(f"❌ Failed to process TIFF image: {e}")

st.sidebar.header("ℹ️ About this App")
st.sidebar.info("This application performs crop type classification and stress detection based on satellite imagery. It uses advanced deep learning models to generate segmentation masks, NDVI statistics, and identify stress causes as biotic or abiotic based on spatial patterns.")
