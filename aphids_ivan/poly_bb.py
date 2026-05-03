import os, json, random, shutil
from pathlib import Path
from PIL import Image

# ── Config ──────────────────────────────────────────
SRC_IMAGES  = Path(r"C:\Users\sudanb\Desktop\rtdetr\aphids_ivan\train_resized\images")
SRC_LABELS  = Path(r"C:\Users\sudanb\Desktop\rtdetr\aphids_ivan\train_resized\labels")
OUT_DIR     = Path(r"C:\Users\sudanb\Desktop\rtdetr\aphids_ivan\dataset")
TARGET_W    = 640
TARGET_H    = 640
SPLIT_RATIO = 0.8
SEED        = 42
CLASS_NAMES = ["aphid"]
# ────────────────────────────────────────────────────

def polygon_to_bbox(coords, img_w, img_h):
    xs = [c * img_w for c in coords[0::2]]
    ys = [c * img_h for c in coords[1::2]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w, h = x_max - x_min, y_max - y_min
    return [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)]

def build_coco(image_list, split_name):
    images_out = OUT_DIR / "images" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    ann_dir = OUT_DIR / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    skipped = 0

    for img_id, img_path in enumerate(sorted(image_list), 1):
        shutil.copy(img_path, images_out / img_path.name)
        print(f"[{img_id}/{len(image_list)}] {img_path.name}")

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": TARGET_W,
            "height": TARGET_H
        })

        label_path = SRC_LABELS / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        with open(label_path) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 7:
                    continue
                class_id = int(parts[0])
                coords = parts[1:]
                bbox = polygon_to_bbox(coords, TARGET_W, TARGET_H)
                area = bbox[2] * bbox[3]
                if area < 1:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,  # fixed: 0-indexed
                    "bbox": bbox,
                    "area": round(area, 2),
                    "iscrowd": 0
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n}  # fixed: 0-indexed
                       for i, n in enumerate(CLASS_NAMES)]
    }

    out_json = ann_dir / f"instances_{split_name}.json"
    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\n[{split_name}] {len(images)} images | "
          f"{len(annotations)} annotations | "
          f"{skipped} missing labels")

# ── Main ─────────────────────────────────────────────
all_images = sorted(SRC_IMAGES.glob("*.jpg"))
random.seed(SEED)
random.shuffle(all_images)

print(f"Found {len(all_images)} images")
if len(all_images) == 0:
    print("ERROR: No images found!")
    exit()

split_idx  = int(len(all_images) * SPLIT_RATIO)
train_imgs = all_images[:split_idx]
val_imgs   = all_images[split_idx:]

print(f"Total: {len(all_images)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")

build_coco(train_imgs, "train")
build_coco(val_imgs,   "val")

print("\nDone! Dataset ready for RT-DETR training.")