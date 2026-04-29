import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# ----------------------------------
# Fix import path for src/
# ----------------------------------
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from model import get_model  # must exist in src/model.py

# ----------------------------------
# Streamlit Page Config
# ----------------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict **Normal** or **Pneumonia**.")

# ----------------------------------
# Device (CPU ONLY)
# ----------------------------------
device = torch.device("cpu")

# ----------------------------------
# Load Model (SAFE)
# ----------------------------------
@st.cache_resource
def load_model():
    model = get_model(num_classes=2)
    model_path = BASE_DIR / "pneumonia_model.pth"

    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    else:
        st.warning("⚠️ Model weights not found. Using untrained model.")

    model.to(device)
    model.eval()
    return model


model = load_model()

# ----------------------------------
# Image Transform
# ----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------
# File Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------
# Prediction
# ----------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            prediction = torch.argmax(outputs, dim=1).item()

        classes = ["Normal", "Pneumonia"]

        st.subheader("🧪 Prediction Result")
        st.success(f"**Result:** {classes[prediction]}")

    except Exception as e:
        st.error("❌ Failed to process image")
        st.exception(e)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("AI Pneumonia Detection | Streamlit + PyTorch")
