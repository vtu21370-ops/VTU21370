import torch
import numpy as np
import cv2
from model_utils import load_model, get_transform

def test_model_inference_on_dummy_input():
    model = load_model()
    model.eval()
    dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    transform = get_transform()
    img_tensor = transform(image=dummy_img)['image'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
    
    assert "boxes" in outputs[0]
    assert "labels" in outputs[0]
    assert isinstance(outputs[0]["boxes"], torch.Tensor)
