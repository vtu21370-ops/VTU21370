import torch
from model_utils import load_model, get_transform
from PIL import Image
import numpy as np
import os

def test_full_pipeline_on_real_image():
    model = load_model()
    model.eval()

    test_img_path = "test_images/example.jpg"
    assert os.path.exists(test_img_path)

    image = Image.open(test_img_path).convert("RGB")
    img_np = np.array(image)
    transform = get_transform()
    img_tensor = transform(image=img_np)['image'].unsqueeze(0)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    assert preds["boxes"].shape[1] == 4  # x1, y1, x2, y2
    assert len(preds["labels"]) == preds["boxes"].shape[0]
    assert all(preds["scores"] <= 1.0)
