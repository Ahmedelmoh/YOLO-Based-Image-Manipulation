# YOLO-Based Image Manipulation Pipeline

A Python pipeline that performs four image manipulation operations using a **single YOLO segmentation model** (YOLOv8-seg). The model runs once per image and its segmentation masks are reused across all sub-tasks.

---

## Project Structure

```
yolo_image_manipulation/
├── main.py                     # Main script — run this
├── models/
│   └── yolov8n-seg.pt          # Auto-downloaded on first run
├── inputs/
│   ├── test_image.jpg          # Your main scene image
│   ├── replacement_object.png  # Object image for sub-task 1B
│   └── new_background.jpg      # Background image for sub-task 1D
├── outputs/
│   ├── object_removed.jpg      # Result of sub-task 1A
│   ├── object_replaced.jpg     # Result of sub-task 1B
│   ├── bg_removed.png          # Result of sub-task 1C (transparent)
│   └── bg_replaced.jpg         # Result of sub-task 1D
└── README.md
```

---

## Installation

```bash
pip install ultralytics opencv-python numpy pillow
```

> Python 3.10+ is required (uses `str | None` type hints).

---

## Quick Start

1. Place your images in the `inputs/` folder:
   - `test_image.jpg` — the scene to manipulate
   - `replacement_object.png` — the replacement object (used in 1B)
   - `new_background.jpg` — the new background (used in 1D)

2. *(Optional)* Open `main.py` and edit the configuration block at the top:

```python
# ── CONFIGURATION ──────────────────────────────────────
TEST_IMAGE        = "inputs/test_image.jpg"
REPLACEMENT_IMAGE = "inputs/replacement_object.png"
NEW_BACKGROUND    = "inputs/new_background.jpg"
MODEL_PATH        = "models/yolov8n-seg.pt"

TARGET_CLASS = None   # e.g. "person", "car" — None = auto-pick first detected
FEATHER_PX   = 11     # Edge softness for background tasks (higher = softer)
```

3. Run:

```bash
py main.py
```

All 4 outputs are saved automatically to the `outputs/` folder. No prompts.

---

## Sub-tasks

### 1A — Object Removal

Removes a selected object class from the scene and fills the gap seamlessly using inpainting.

**Pipeline:**
1. Run YOLO segmentation on the input image
2. Build a binary mask for the target class (all instances combined)
3. Dilate the mask slightly to cover hard pixel edges
4. Apply `cv2.inpaint` with the **Telea** method to fill the removed region
5. Save result to `outputs/object_removed.jpg`

**Key call:**
```python
cv2.inpaint(img, mask_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
```

---

### 1B — Object Replacement

Replaces a detected object with a custom image, blended naturally into the scene.

**Pipeline:**
1. Build mask for target class → extract bounding box `(x1, y1, x2, y2)`
2. Resize the replacement image to fit the bounding box
3. Paste it onto a copy of the original scene
4. Apply `cv2.seamlessClone` (Poisson blending) for natural color/lighting match
5. Falls back to alpha blending if seamlessClone fails (e.g. mask near image border)
6. Save result to `outputs/object_replaced.jpg`

**Key call:**
```python
cv2.seamlessClone(src_patch, img, clone_mask, center, cv2.NORMAL_CLONE)
```

---

### 1C — Background Removal

Keeps all detected foreground subjects and makes the background fully transparent.

**Pipeline:**
1. Build a union mask of **all** detected object masks
2. Apply Gaussian blur to mask edges (feathering) for a smooth cutout
3. Use the feathered mask as the alpha channel of an RGBA image
4. Save result as a transparent `outputs/bg_removed.png`

**Key call:**
```python
rgba = cv2.merge([b, g, r, alpha_channel])
```

---

### 1D — Background Replacement

Composites the foreground subjects onto a completely new background.

**Pipeline:**
1. Extract the same feathered foreground mask used in 1C
2. Resize the new background image to match the original dimensions
3. Alpha-composite: `foreground × α + new_background × (1 − α)`
4. Save result to `outputs/bg_replaced.jpg`

**Key call:**
```python
result = (img * alpha + new_bg * (1 - alpha)).clip(0, 255).astype(np.uint8)
```

---

## How the Pipeline Works Internally

```
test_image.jpg
      │
      ▼
 YOLO runs ONCE
      │
      ├─── masks + bounding boxes ───► 1A: inpaint removed region
      │
      ├─── masks + bounding boxes ───► 1B: seamlessClone replacement
      │
      ├─── union of all masks ────────► 1C: transparent RGBA output
      │
      └─── union of all masks ────────► 1D: composite onto new background
```

YOLO inference runs **once** and all 4 tasks share the same `results` object — no redundant computation.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `TEST_IMAGE` | `inputs/test_image.jpg` | Main scene image |
| `REPLACEMENT_IMAGE` | `inputs/replacement_object.png` | Object used in sub-task 1B |
| `NEW_BACKGROUND` | `inputs/new_background.jpg` | Background used in sub-task 1D |
| `MODEL_PATH` | `models/yolov8n-seg.pt` | YOLO model (auto-downloaded) |
| `TARGET_CLASS` | `None` | Class to remove/replace (`None` = first detected) |
| `FEATHER_PX` | `11` | Gaussian blur radius for mask edge smoothing |

---

## Supported YOLO Models

Any YOLOv8 or YOLOv11 segmentation model works. Swap `MODEL_PATH` to change:

| Model | Speed | Accuracy |
|---|---|---|
| `yolov8n-seg.pt` | ⚡ Fastest | Good |
| `yolov8s-seg.pt` | Fast | Better |
| `yolov8m-seg.pt` | Medium | Best for most uses |
| `yolov11n-seg.pt` | ⚡ Fastest | Good |

---

## Example Output

```
╔══════════════════════════════════════════════════╗
║    YOLO-Based Image Manipulation Pipeline        ║
║    Running all 4 sub-tasks automatically …       ║
╚══════════════════════════════════════════════════╝

[INFO] Loading model: models/yolov8n-seg.pt
[INFO] Detected objects:
       [0] person
       [2] car
[INFO] Target class for 1A & 1B: 'person'

── Sub-task 1A — Object Removal
  ✓ [1A Object Removed]              →  outputs\object_removed.jpg

── Sub-task 1B — Object Replacement
  ✓ [1B Object Replaced]             →  outputs\object_replaced.jpg

── Sub-task 1C — Background Removal
  ✓ [1C Background Removed (RGBA)]   →  outputs\bg_removed.png

── Sub-task 1D — Background Replacement
  ✓ [1D Background Replaced]         →  outputs\bg_replaced.jpg
```

---

## Notes

- The `outputs/` folder is created automatically if it does not exist.
- The YOLO model is downloaded automatically on first run (~6 MB for `yolov8n-seg.pt`).
- Sub-tasks 1A and 1B operate on the **target class only**. Sub-tasks 1C and 1D operate on **all detected objects** as the foreground.
- For best inpainting results in 1A, use images where the background behind the object is relatively uniform.
- For best blending in 1B, use a replacement image with a transparent background (PNG with alpha channel).