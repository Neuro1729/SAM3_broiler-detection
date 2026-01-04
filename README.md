# 🐔 SAM3_broiler-detection  

High-resolution **broiler (chicken) detection and weight estimation** using **SAM-3 with tiling** and **DPT Transformer depth estimation** for dense poultry scenes and large farm images.

This project demonstrates how **tiled SAM-3 segmentation + depth-aware size estimation + hallucination control** produces far more reliable chicken measurements than full-image or YOLO-based pipelines.

---

## 🔬 Why Tiling?

SAM models struggle when:
- Images are very large  
- Many small chickens appear close together  

By dividing the image into smaller tiles (3×3, 4×4, etc.):

- Each chicken becomes larger in pixel space  
- SAM-3 detects boundaries more accurately  
- GPU memory usage is reduced  
- Overlapping chickens are separated better  

---

## 🎥 Demo Videos

### 3 × 3 Tiling – SAM-3 Detection
[![3 x 3 Tiling SAM](https://img.youtube.com/vi/dKcmJSND6qQ/0.jpg)](https://youtu.be/dKcmJSND6qQ)

### 4 × 4 Tiling – SAM-3 Detection
[![4 x 4 Tiling SAM](https://img.youtube.com/vi/YA9dF6aVD7Y/0.jpg)](https://youtu.be/YA9dF6aVD7Y)

### YOLO-Based Baseline
[![YOLO Baseline](https://img.youtube.com/vi/J-06fdpUQFU/0.jpg)](https://youtu.be/J-06fdpUQFU)

---

## 🧠 Weight Estimation Using DPT Transformer

We do **not** estimate weight from pixel area alone.

We use:

**DPT Transformer (Depth Prediction Transformer)**  
to estimate how far each part of the chicken is from the camera.

### Pipeline
1. SAM-3 segments each chicken  
2. DPT predicts a depth map  
3. Depth values inside each chicken mask are integrated  
4. This produces a **distance-aware size estimate**  
5. This is mapped to chicken weight  

This corrects for:
- Camera distance  
- Perspective distortion  
- Birds appearing smaller when farther away  

---

## ⚙️ Full Pipeline

1. Image is split into tiles (3×3 or 4×4)  
2. Each tile runs through **SAM-3**  
3. Masks are merged using IoU filtering  
4. **DPT Transformer** predicts depth  
5. Mask + depth → **distance-corrected size**  
6. Size → **weight estimation**

---

## 🛡️ Hallucination & Outlier Control

Large vision models sometimes **hallucinate objects**, produce **ghost masks**, or generate **wrong bounding boxes**.

We handle this by:

- Removing masks with abnormal area or shape  
- Filtering out depth-inconsistent detections  
- Using IoU-based merging across tiles  
- Rejecting outliers that do not match poultry geometry  

This prevents:
- False chickens  
- Floating masks  
- Broken bounding boxes  
- Incorrect weight estimates  

---

## 🔑 Model Access (No Manual Weight Download)

You do **not** need to manually download weights.  
SAM-3 and DPT models are pulled automatically from Hugging Face after login.

Run this once:

```python
!pip install git+https://github.com/huggingface/transformers.git
!pip install huggingface_hub --upgrade

from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")
```
# 🔮 Improvements & Roadmap

## 1️⃣ Temporal Tracking with SAM-3 Video

**Goal:** Enable robust, identity-preserving chicken tracking across video frames with minimal GPU usage.

### Usage
```python
from transformers import Sam3VideoModel, Sam3VideoProcessor
```

### Pipeline
1. Run **tiled SAM-3** on the first frame  
2. Extract the **best segmentation masks**  
3. **Track all chickens** using **SAM-3 Video**

### Benefits
- ✅ Seamless, temporally consistent video output  
- ✅ Stable chicken identities across frames  
- ✅ Much lower GPU memory and compute cost  
- ✅ Enables real-time monitoring on modest hardware (e.g., Colab T4)

---

## 2️⃣ Mixture of Prompts

**Goal:** Improve detection robustness under varying farm conditions.

### Prompt Ensemble
```python
prompts = ["broiler", "chicken", "fowl", "poultry"]
```

### Improvements
- 🛡️ Higher detection robustness across breeds and sizes  
- 👁️ Better handling of partial occlusions  
- 🌞 Enhanced recall under diverse lighting and camera angles  

---

## 3️⃣ Fast & Quantized SAM-3

**Goal:** Optimize for real-world edge deployment on farm cameras.

### Techniques
- Use **FAST-SAM** for speed  
- Apply **quantized SAM-3** (INT8 or 4-bit)  
- Combine with **tiling** for high-resolution inputs

### Capabilities
- 📦 Runs on **edge devices** (e.g., Jetson, Raspberry Pi with Coral)  
- 📹 Enables **real-time processing** on live farm cameras  
- 💰 Supports **low-cost GPU deployment** (ideal for Colab or budget cloud instances)  
- 🎥 Output compatible with `.mp4` (H.264) for small file sizes and broad compatibility
