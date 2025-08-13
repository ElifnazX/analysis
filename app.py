import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# Başlık
st.title("🫁 X-ray Hastalık Tanı Sistemi")

# Sınıf isimleri (model sırasına göre olmalı)
classes = ["COVID-19", "Normal", "Pneumonia"]

# Modeli yükle
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Görüntü tahmini yapan fonksiyon
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

# Görüntü yükleme arayüzü
uploaded_file = st.file_uploader("🖼️ X-ray Görüntüsü Yükleyin", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
    prediction = predict(image)
    st.success(f"🧠 Model Tahmini: **{prediction}**")
    
# streamlit run app.py
#C:\msys64\ucrt64\bin