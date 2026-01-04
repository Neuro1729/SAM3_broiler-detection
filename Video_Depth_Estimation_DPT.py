import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# ------------------------------
# 1️⃣ Load DPT model
# ------------------------------
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# 2️⃣ Open video and read first frame
# ------------------------------
video_path = "/content/2025_12_15_15_24_16_4_dQCiGf.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("Could not open video")

ret, frame = cap.read()
cap.release()  # Close immediately after reading first frame

if not ret:
    raise Exception("Could not read first frame")

# ------------------------------
# 3️⃣ Estimate depth for first frame
# ------------------------------
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

inputs = feature_extractor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()  # (H, W)

# ------------------------------
# 4️⃣ Save depth map as .npy
# ------------------------------
output_path = "/content/first_frame_depth.npy"
np.save(output_path, predicted_depth)
print(f"✅ First-frame depth map saved: {output_path}")
print(f"Depth shape: {predicted_depth.shape}")
