# 🐔 SAM3_broiler-detection  

High-resolution **broiler (chicken) detection** using **SAM-3 with tiling** to overcome GPU memory limits and improve segmentation accuracy on large poultry images.

This project demonstrates how splitting large images into tiles (3×3, 4×4, etc.) allows **SAM-3** to detect small, dense chickens much more accurately than running it on the full image at once.

---

## 🔬 Why Tiling?

SAM models struggle when many small objects are present in a large image.  
By dividing the image into smaller tiles:

- Each chicken becomes larger in pixel space  
- SAM can focus on fine details  
- GPU memory usage is reduced  
- Detection accuracy increases  

---

## 🎥 Demo Videos

### 3 × 3 Tiling – SAM-3 Detection
[![3 x 3 Tiling SAM](https://img.youtube.com/vi/dKcmJSND6qQ/0.jpg)](https://youtu.be/dKcmJSND6qQ)

### 4 × 4 Tiling – SAM-3 Detection
[![4 x 4 Tiling SAM](https://img.youtube.com/vi/J-06fdpUQFU/0.jpg)](https://youtu.be/J-06fdpUQFU)

---

## ⚙️ How It Works

1. The input poultry image is split into tiles (3×3 or 4×4).
2. Each tile is passed independently through **SAM-3**.
3. Masks from all tiles are merged back into the full image.
4. Overlapping regions are merged using IoU-based filtering.
5. Final segmentation gives accurate chicken boundaries even in dense flocks.

---

## 🧠 Model Weights (Required)

This project uses **Meta’s SAM-3** weights.

Because of licensing and size, **you must download the weights manually from Hugging Face**.

### Steps:
1. Create a Hugging Face account  
   👉 https://huggingface.co/join  

2. Accept Meta’s SAM-3 license on Hugging Face  
3. Download the SAM-3 weights  
4. Place them in your project folder (for example):
