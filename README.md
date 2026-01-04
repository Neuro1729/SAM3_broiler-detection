# 🐔 SAM3_broiler-detection  

High-resolution **broiler (chicken) detection and weight estimation** using **SAM-3 with tiling** and **DPT Transformer depth estimation** for dense poultry scenes and large farm images.

This project demonstrates how **tiled SAM-3 segmentation + depth-aware area estimation** produces much more reliable chicken size and weight proxies than full-image segmentation alone.

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
[![4 x 4 Tiling SAM](https://img.youtube.com/vi/J-06fdpUQFU/0.jpg)](https://youtu.be/J-06fdpUQFU)

---

## 🧠 Weight Estimation Using DPT Transformer

We do **not** estimate weight from pixel area alone.

We use:

**DPT Transformer (Depth Prediction Transformer)**  
to generate a **depth map** of the scene.

### Pipeline
1. SAM-3 segments each chicken  
2. DPT predicts depth  
3. Depth is integrated inside each chicken mask  
4. This gives a **volume-aware area proxy**  
5. That proxy is mapped to chicken weight  

This compensates for:
- Camera distance  
- Perspective distortion  
- Body thickness  

---

## ⚙️ Full Pipeline

1. Image is split into tiles (3×3 or 4×4)  
2. Each tile runs through **SAM-3**  
3. Masks are merged using IoU filtering  
4. **DPT Transformer** predicts depth  
5. Mask × depth → **volume proxy**  
6. Volume proxy → **weight estimation**

---

## 🔑 Model Access (No Manual Weight Download)

You do **not** need to manually download weights.  
SAM-3 and DPT models are pulled automatically from Hugging Face after login.

Run this once in Colab or your environment:

```python
!pip install git+https://github.com/huggingface/transformers.git
!pip install huggingface_hub --upgrade

from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")
