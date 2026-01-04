# 🐔 SAM3_broiler-detection  

High-resolution **broiler (chicken) detection and weight estimation** using **SAM-3 with tiling** and **DPT Transformer depth estimation** for dense poultry scenes and large farm images.

This project demonstrates how **tiled SAM-3 segmentation + depth-aware size estimation** produces more reliable chicken weight proxies than full-image or YOLO-based methods.

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
3. The depth values inside each chicken mask are averaged and integrated  
4. This produces a **distance-aware size estimate**  
5. This is mapped to chicken weight  

This corrects for:
- Camera distance  
- Perspective distortion  
- Birds appearing smaller when farther away  

So two chickens with the same pixel area but different distances do **not** get the same weight.

---

## ⚙️ Full Pipeline

1. Image is split into tiles (3×3 or 4×4)  
2. Each tile runs through **SAM-3**  
3. Masks are merged using IoU filtering  
4. **DPT Transformer** predicts depth  
5. Mask + depth → **distance-corrected size**  
6. Size → **weight estimation**

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
