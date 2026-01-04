from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import label
import gc
import json
import zipfile
from io import BytesIO

# Import SAM-3 (kept globally for efficiency)
from transformers import Sam3Model, Sam3Processor

app = FastAPI(title="Chicken Weight Proxy Analyzer", version="1.0")

# ------------------ UTILITY FUNCTIONS ------------------
def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compute_intersection_area(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

def nms_with_containment(boxes, scores, iou_threshold=0.2):
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while sorted_indices:
        best = sorted_indices.pop(0)
        keep.append(best)
        remaining = []
        for idx in sorted_indices:
            b1, b2 = boxes[best], boxes[idx]
            area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area_b2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            if area_b1 <= 0 or area_b2 <= 0:
                remaining.append(idx)
                continue
            inter_area = compute_intersection_area(b1, b2)
            if inter_area / area_b2 >= 0.9 or inter_area / area_b1 >= 0.9:
                continue
            iou = compute_iou(b1, b2)
            if iou < iou_threshold:
                remaining.append(idx)
        sorted_indices = remaining
    return keep

# ------------------ GLOBAL: LOAD SAM-3 ONCE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

try:
    sam_processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam_model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    if device.type == "cuda":
        sam_model = sam_model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32
    print("✅ SAM-3 model loaded globally")
except Exception as e:
    sam_model = None
    print(f"⚠️ SAM-3 failed to load: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "sam3_loaded": sam_model is not None
    }

# ------------------ MAIN ENDPOINT ------------------
@app.post("/analyze_video")
async def analyze_video(video_file: UploadFile = File(...)):
    if sam_model is None:
        raise HTTPException(status_code=500, detail="SAM-3 model not loaded")

    if video_file.content_type != "video/mp4":
        raise HTTPException(status_code=400, detail="Only MP4 videos supported")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save input video
        video_path = os.path.join(tmp_dir, "input.mp4")
        with open(video_path, "wb") as f:
            f.write(await video_file.read())

        # Read first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")
        ret, first_frame = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read first frame")
        H, W = first_frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # -------------------------------------------------
        # ✅ LOAD DPT → ESTIMATE DEPTH → UNLOAD IMMEDIATELY
        # -------------------------------------------------
        print("📦 Loading DPT for first-frame depth...")
        from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        dpt_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device).eval()

        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(first_frame_rgb)

        inputs = dpt_extractor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = dpt_model(**inputs)
            depth_small = outputs.predicted_depth.squeeze().cpu().numpy()

        depth_full = cv2.resize(depth_small, (W, H), interpolation=cv2.INTER_CUBIC)

        # 🔥 UNLOAD DPT TO FREE MEMORY
        del dpt_model, dpt_extractor, inputs, outputs, depth_small, image_pil, first_frame_rgb, first_frame
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 DPT model unloaded — VRAM freed for SAM-3")

        # ------------------ SAM-3 PIPELINE ------------------
        TILE_ROWS, TILE_COLS = 3, 3
        tile_w = W // TILE_COLS
        tile_h = H // TILE_ROWS
        overlap_w = int(tile_w * 0.3)
        overlap_h = int(tile_h * 0.3)
        tile_w += overlap_w
        tile_h += overlap_h

        tiles = []
        for row in range(TILE_ROWS):
            for col in range(TILE_COLS):
                x_start = max(0, col * (W // TILE_COLS) - overlap_w // 2)
                y_start = max(0, row * (H // TILE_ROWS) - overlap_h // 2)
                x_end = min(W, x_start + tile_w)
                y_end = min(H, y_start + tile_h)
                if x_end > x_start and y_end > y_start:
                    tiles.append({'x_start': x_start, 'y_start': y_start, 'x_end': x_end, 'y_end': y_end})

        MIN_WIDTH, MIN_HEIGHT = 0, 0
        MAX_WIDTH, MAX_HEIGHT = W // 4, H // 4
        MIN_ASPECT, MAX_ASPECT = 0.4, 2.5

        next_track_id = 0
        active_tracks = {}
        iou_match_thresh = 0.3
        max_age = 5
        detections_log = []

        cap = cv2.VideoCapture(video_path)
        output_video_path = os.path.join(tmp_dir, "output.mp4")
        # Use H.264 for smaller file size (requires ffmpeg support)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'mp4v' if 'avc1' fails
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

        frame_count = 0
        max_frames = 200

        while frame_count < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            all_masks = []
            for tile_info in tiles:
                x_start, y_start, x_end, y_end = tile_info['x_start'], tile_info['y_start'], tile_info['x_end'], tile_info['y_end']
                tile_pil = image_pil.crop((x_start, y_start, x_end, y_end))

                inputs = sam_processor(images=tile_pil, text="Fowl", return_tensors="pt")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device=device, dtype=dtype if v.dtype == torch.float32 else v.dtype)

                with torch.no_grad():
                    outputs = sam_model(**inputs)

                results = sam_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.3,
                    mask_threshold=0.3,
                    target_sizes=inputs["original_sizes"].tolist(),
                )[0]

                for mask_tensor, score in zip(results["masks"], results["scores"]):
                    local_mask = (mask_tensor.cpu().numpy() > 0.5)
                    labeled_mask, num_features = label(local_mask)
                    if num_features > 0:
                        component_sizes = np.bincount(labeled_mask.ravel())
                        largest_label = component_sizes[1:].argmax() + 1
                        local_mask = (labeled_mask == largest_label)

                    h_local, w_local = local_mask.shape
                    full_mask = np.zeros((H, W), dtype=bool)
                    h_eff = min(h_local, H - y_start)
                    w_eff = min(w_local, W - x_start)
                    full_mask[y_start:y_start+h_eff, x_start:x_start+w_eff] = local_mask[:h_eff, :w_eff]

                    if not np.any(full_mask):
                        continue

                    rows = np.any(full_mask, axis=1)
                    cols = np.any(full_mask, axis=0)
                    if not rows.any() or not cols.any():
                        continue
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]

                    width, height = x_max - x_min, y_max - y_min
                    if width < MIN_WIDTH or height < MIN_HEIGHT: continue
                    if width > MAX_WIDTH or height > MAX_HEIGHT: continue
                    if not (MIN_ASPECT <= width / height <= MAX_ASPECT): continue

                    all_masks.append({
                        'mask': full_mask,
                        'score': score.cpu().item(),
                        'bbox': (x_min, y_min, x_max, y_max)
                    })

                del inputs, outputs, results, tile_pil
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # NMS
            final_masks = []
            if all_masks:
                keep_indices = nms_with_containment(
                    [m['bbox'] for m in all_masks],
                    [m['score'] for m in all_masks],
                    iou_threshold=0.2
                )
                final_masks = [all_masks[i] for i in keep_indices]

            # Tracking
            current_dets = final_masks.copy()
            unmatched = list(range(len(current_dets)))

            for track_id in list(active_tracks.keys()):
                track = active_tracks[track_id]
                best_iou = iou_match_thresh
                best_idx = -1
                for i in unmatched:
                    iou = compute_iou(track['bbox'], current_dets[i]['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_idx != -1:
                    active_tracks[track_id]['bbox'] = current_dets[best_idx]['bbox']
                    active_tracks[track_id]['confidence'] = current_dets[best_idx]['score']
                    active_tracks[track_id]['last_frame'] = frame_count
                    unmatched.remove(best_idx)
                else:
                    if frame_count - active_tracks[track_id]['last_frame'] > max_age:
                        del active_tracks[track_id]

            for i in unmatched:
                active_tracks[next_track_id] = {
                    'bbox': current_dets[i]['bbox'],
                    'confidence': current_dets[i]['score'],
                    'last_frame': frame_count
                }
                next_track_id += 1

            # Weight proxy using depth_full
            frame_detections = []
            for track_id, track in active_tracks.items():
                x1, y1, x2, y2 = track['bbox']
                best_mask = None
                for m in final_masks:
                    if compute_iou(m['bbox'], (x1, y1, x2, y2)) > 0.9:
                        best_mask = m['mask']
                        break
                if best_mask is not None:
                    mask = best_mask
                else:
                    mask = np.zeros((H, W), dtype=bool)
                    mask[y1:y2, x1:x2] = True

                mask_depth_values = depth_full[mask]
                if mask_depth_values.size == 0:
                    median_depth = 1.0
                    weight_proxy = 0.0
                    mask_area = 0
                else:
                    median_depth = np.median(mask_depth_values)
                    mask_area = np.sum(mask)
                    weight_proxy = float(mask_area / (median_depth ** 2 + 1e-6))

                frame_detections.append({
                    "id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(track['confidence']),
                    "mask_area_pixels": int(mask_area),
                    "median_depth": float(median_depth),
                    "weight_proxy": weight_proxy
                })

            detections_log.append({
                "frame": int(frame_count),
                "timestamp_sec": float(frame_count) / fps,
                "count": len(frame_detections),
                "tracks": frame_detections
            })

            # Draw and write frame
            display_frame = frame_bgr.copy()
            if final_masks:
                combined_mask = np.zeros((H, W), dtype=bool)
                for item in final_masks:
                    combined_mask |= item['mask']
                overlay = np.zeros_like(display_frame, dtype=np.uint8)
                overlay[combined_mask] = [0, 255, 0]
                display_frame = cv2.addWeighted(display_frame, 0.55, overlay, 0.45, 0)
                for item in final_masks:
                    x_min, y_min, x_max, y_max = item['bbox']
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            out.write(display_frame)
            frame_count += 1

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        cap.release()
        out.release()

        # Save JSON
        json_output = {
            "video_info": {
                "filename": video_file.filename,
                "duration_sec": frame_count / fps,
                "fps": fps,
                "width": W,
                "height": H
            },
            "depth_convention_note": (
                "Depth from DPT (monocular): higher = closer. "
                "Weight proxy = mask_area_pixels / (median_depth² + 1e-6). Relative units."
            ),
            "counts_over_time": [
                {"timestamp_sec": d["timestamp_sec"], "count": d["count"]}
                for d in detections_log
            ],
            "weight_estimates": {
                "per_bird": [
                    {
                        "track_id": t["id"],
                        "weight_proxy": t["weight_proxy"],
                        "confidence": t["confidence"]
                    }
                    for d in detections_log for t in d["tracks"]
                ],
                "aggregate": {
                    "total_weight_proxy": sum(t["weight_proxy"] for d in detections_log for t in d["tracks"])
                }
            }
        }

        json_path = os.path.join(tmp_dir, "detections.json")
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)

        # ------------------ RETURN ZIP ------------------
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_video_path, "output.mp4")
            zf.write(json_path, "detections.json")
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=chicken_analysis.zip"}
        )
