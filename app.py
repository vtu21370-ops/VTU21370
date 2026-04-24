import streamlit as st
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import torchvision

# Class names for multi-class detection
CLASS_NAMES = [
    "black_core", "corner", "crack", "finger", "fragment", "horizontal_dislocation",
    "printing_error", "scratch", "short_circuit", "star_crack", "thick_line", "vertical_dislocation"
]

# Load trained model
@st.cache_resource
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(CLASS_NAMES) + 1)
    model.load_state_dict(torch.load("solar_defect_trained_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Inference function
def predict(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return predictions

def visualize_predictions(image, predictions):
    image = np.array(image)
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{CLASS_NAMES[label-1]}: {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Streamlit UI
st.title("Solar Panel Defect Detection")
st.write("Upload an image to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    predictions = predict(image)
    result_image = visualize_predictions(image, predictions)
    
    st.image(result_image, caption="Detected Defects", use_column_width=True)
