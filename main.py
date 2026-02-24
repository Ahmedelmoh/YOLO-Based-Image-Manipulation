"""
YOLO-Based Image Manipulation Pipeline
========================================
Sub-tasks:
  1A - Object Removal
  1B - Object Replacement
  1C - Background Removal
  1D - Background Replacement

"""

import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════
# 1-  CONFIGURATION 
# ══════════════════════════════════════════════════════════════════

# Input files
TEST_IMAGE        = "F:\\NTI\\Technical\\Evaluation_tasks\\YOLO-Based Image Manipulation\\Inputs\\image_2.jpg"
REPLACEMENT_IMAGE = "F:\\NTI\\Technical\\Evaluation_tasks\\YOLO-Based Image Manipulation\\Inputs\\replace_1.jpeg"
NEW_BACKGROUND    = "F:\\NTI\\Technical\\Evaluation_tasks\\YOLO-Based Image Manipulation\\Inputs\\background.jpg"

# YOLO segmentation model (auto-downloads if not present)
MODEL_PATH        = "models/yolov8n-seg.pt"

# Output files
OUT_OBJECT_REMOVED  = "outputs/object_removed.jpg"
OUT_OBJECT_REPLACED = "outputs/object_replaced.jpg"
OUT_BG_REMOVED      = "outputs/bg_removed.png"
OUT_BG_REPLACED     = "outputs/bg_replaced.jpg"


TARGET_CLASS = "person"         

# Mask edge feathering radius (pixels) — higher = softer edges
FEATHER_PX = 11


# ══════════════════════════════════════════════════════════════════
# 2-  YOLO HELPERS
# ══════════════════════════════════════════════════════════════════

def load_model() -> YOLO:
    print(f"[INFO] Loading model: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def run_segmentation(model: YOLO, img: np.ndarray):
    return model(img, verbose=False)


def get_detected_classes(results) -> dict:
    """Return {class_id: class_name} for every detected object."""
    names    = results[0].names
    detected = {}
    if results[0].boxes is not None:
        for cls_id in results[0].boxes.cls.cpu().numpy().astype(int):
            detected[int(cls_id)] = names[int(cls_id)]
    return detected


def resolve_target_class(detected: dict) -> str:
    """
    Return TARGET_CLASS if it exists in detections,
    otherwise fall back to the first detected class.
    """
    if TARGET_CLASS and TARGET_CLASS.lower() in [v.lower() for v in detected.values()]:
        return TARGET_CLASS
    first = list(detected.values())[0]
    if TARGET_CLASS:
        print(f"[WARN] '{TARGET_CLASS}' not detected — using '{first}' instead.")
    return first


def build_mask_for_class(results, target_class: str, shape: tuple) -> np.ndarray:
    """Binary uint8 mask (0/255) for all instances of target_class."""
    h, w     = shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    if results[0].masks is None:
        return combined
    masks_data = results[0].masks.data.cpu().numpy()
    classes    = results[0].boxes.cls.cpu().numpy().astype(int)
    names      = results[0].names
    for i, cls_id in enumerate(classes):
        if names[int(cls_id)].lower() == target_class.lower():
            m         = masks_data[i]
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            combined  = np.maximum(combined, (m_resized > 0.5).astype(np.uint8) * 255)
    return combined


def build_all_foreground_mask(results, shape: tuple) -> np.ndarray:
    """Union of ALL detected object masks."""
    h, w     = shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    if results[0].masks is None:
        return combined
    for m in results[0].masks.data.cpu().numpy():
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        combined  = np.maximum(combined, (m_resized > 0.5).astype(np.uint8) * 255)
    return combined


def feather_mask(mask: np.ndarray) -> np.ndarray:
    """Gaussian-blur mask edges → smooth float32 alpha in [0, 1]."""
    k       = FEATHER_PX | 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return blurred / 255.0


def save_image(img: np.ndarray, path: str, label: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"  ✓ [{label}]  →  {os.path.abspath(path)}")


# ══════════════════════════════════════════════════════════════════
# 3-  SUB-TASK 1A — Object Removal
# ══════════════════════════════════════════════════════════════════

def task_1a(img: np.ndarray, results, target_class: str):
    """
    Pipeline:
      1. Build binary mask for target_class
      2. Dilate mask (covers hard pixel edges)
      3. cv2.inpaint with Telea method fills the hole seamlessly
      4. Save outputs/object_removed.jpg
    """
    mask = build_mask_for_class(results, target_class, img.shape)
    if mask.max() == 0:
        print(f"  [WARN 1A] No mask found for '{target_class}' — skipped.")
        return

    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_dilated = cv2.dilate(mask, kernel)
    result       = cv2.inpaint(img, mask_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    save_image(result, OUT_OBJECT_REMOVED, "1A Object Removed")


# ══════════════════════════════════════════════════════════════════
# 4-  SUB-TASK 1B — Object Replacement
# ══════════════════════════════════════════════════════════════════

def task_1b(img: np.ndarray, results, target_class: str, replacement: np.ndarray):
    """
    Pipeline:
      1. Get bounding box of target_class mask
      2. Resize replacement image to that bounding box
      3. cv2.seamlessClone (Poisson blend) for natural edges
         → falls back to alpha blend if seamlessClone fails
      4. Save outputs/object_replaced.jpg
    """
    mask = build_mask_for_class(results, target_class, img.shape)
    if mask.max() == 0:
        print(f"  [WARN 1B] No mask found for '{target_class}' — skipped.")
        return

    ys, xs = np.where(mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1

    rep = cv2.resize(replacement, (bw, bh))

    # Handle RGBA replacement images
    if rep.ndim == 3 and rep.shape[2] == 4:
        alpha_rep = rep[:, :, 3]
        rep_rgb   = rep[:, :, :3]
    else:
        alpha_rep = np.full((bh, bw), 255, dtype=np.uint8)
        rep_rgb   = rep if rep.ndim == 3 else cv2.cvtColor(rep, cv2.COLOR_GRAY2BGR)

    src_patch                  = img.copy()
    src_patch[y1:y2, x1:x2]   = rep_rgb

    clone_mask                  = np.zeros(img.shape[:2], dtype=np.uint8)
    clone_mask[y1:y2, x1:x2]   = alpha_rep
    kernel                      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clone_mask                  = cv2.erode(clone_mask, kernel)

    h, w = img.shape[:2]
    cx   = max(1, min((x1 + x2) // 2, w - 2))
    cy   = max(1, min((y1 + y2) // 2, h - 2))

    try:
        result = cv2.seamlessClone(src_patch, img, clone_mask, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error as e:
        print(f"  [INFO 1B] seamlessClone failed ({e}); using alpha-blend fallback.")
        alpha_f = clone_mask.astype(np.float32)[:, :, None] / 255.0
        result  = (src_patch.astype(np.float32) * alpha_f +
                   img.astype(np.float32) * (1 - alpha_f)).clip(0, 255).astype(np.uint8)

    save_image(result, OUT_OBJECT_REPLACED, "1B Object Replaced")


# ══════════════════════════════════════════════════════════════════
# 5-  SUB-TASK 1C — Background Removal
# ══════════════════════════════════════════════════════════════════

def task_1c(img: np.ndarray, results):
    """
    Pipeline:
      1. Union of all detected object masks = foreground
      2. Feather (Gaussian blur) mask edges for smooth cutout
      3. Build RGBA image — background pixels become transparent
      4. Save outputs/bg_removed.png
    """
    fg_mask = build_all_foreground_mask(results, img.shape)
    if fg_mask.max() == 0:
        print("  [WARN 1C] No foreground detected — output will be fully transparent.")

    alpha_f  = feather_mask(fg_mask)
    b, g, r  = cv2.split(img)
    alpha_ch = (alpha_f * 255).clip(0, 255).astype(np.uint8)
    rgba     = cv2.merge([b, g, r, alpha_ch])

    save_image(rgba, OUT_BG_REMOVED, "1C Background Removed (RGBA PNG)")


# ══════════════════════════════════════════════════════════════════
# 6-  SUB-TASK 1D — Background Replacement
# ══════════════════════════════════════════════════════════════════

def task_1d(img: np.ndarray, results, new_bg: np.ndarray):
    """
    Pipeline:
      1. Same foreground mask as 1C
      2. Resize new background to match original dimensions
      3. Alpha-composite: foreground * alpha + new_bg * (1 - alpha)
      4. Save outputs/bg_replaced.jpg
    """
    h, w   = img.shape[:2]
    new_bg = cv2.resize(new_bg, (w, h))

    fg_mask = build_all_foreground_mask(results, img.shape)
    if fg_mask.max() == 0:
        print("  [WARN 1D] No foreground detected — output will be the new background only.")

    alpha_f   = feather_mask(fg_mask)[:, :, None]
    composite = (img.astype(np.float32) * alpha_f +
                 new_bg.astype(np.float32) * (1.0 - alpha_f))
    result    = composite.clip(0, 255).astype(np.uint8)

    save_image(result, OUT_BG_REPLACED, "1D Background Replaced")


# ══════════════════════════════════════════════════════════════════
# 7-  MAIN 
# ══════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════╗
║    YOLO-Based Image Manipulation Pipeline        ║
║    Running all 4 sub-tasks automatically …       ║
╚══════════════════════════════════════════════════╝""")

    # ── Validate input files ─────────────────────────────────────
    missing = [p for p in (TEST_IMAGE, REPLACEMENT_IMAGE, NEW_BACKGROUND)
               if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"[ERROR] Input file not found: '{p}'")
        sys.exit(1)

    # ── Load images ───────────────────────────────────────────────
    print("\n[INFO] Loading input images …")
    img         = cv2.imread(TEST_IMAGE)
    replacement = cv2.imread(REPLACEMENT_IMAGE, cv2.IMREAD_UNCHANGED)
    new_bg      = cv2.imread(NEW_BACKGROUND)

    # ── Run YOLO once — reused by all tasks ──────────────────────
    model   = load_model()
    results = run_segmentation(model, img)

    detected = get_detected_classes(results)
    if not detected:
        sys.exit("[ERROR] YOLO detected no objects in the test image.")

    print("\n[INFO] Detected objects:")
    for cid, cname in detected.items():
        print(f"       [{cid}] {cname}")

    target = resolve_target_class(detected)
    print(f"\n[INFO] Target class for 1A & 1B: '{target}'\n")

    # ── Sub-task 1A ───────────────────────────────────────────────
    print("─" * 52)
    print("  Sub-task 1A — Object Removal")
    print("─" * 52)
    task_1a(img, results, target)

    # ── Sub-task 1B ───────────────────────────────────────────────
    print("\n" + "─" * 52)
    print("  Sub-task 1B — Object Replacement")
    print("─" * 52)
    task_1b(img, results, target, replacement)

    # ── Sub-task 1C ───────────────────────────────────────────────
    print("\n" + "─" * 52)
    print("  Sub-task 1C — Background Removal")
    print("─" * 52)
    task_1c(img, results)

    # ── Sub-task 1D ───────────────────────────────────────────────
    print("\n" + "─" * 52)
    print("  Sub-task 1D — Background Replacement")
    print("─" * 52)
    task_1d(img, results, new_bg)

    print(f"""
╔══════════════════════════════════════════════════╗
║  Done! All outputs saved:                        ║
║                                                  ║
║  1A  {OUT_OBJECT_REMOVED:<42}║
║  1B  {OUT_OBJECT_REPLACED:<42}║
║  1C  {OUT_BG_REMOVED:<42}║
║  1D  {OUT_BG_REPLACED:<42}║
╚══════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()