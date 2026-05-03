# resize_images.py
from pathlib import Path
from PIL import Image

SRC = Path(r"C:\Users\sudanb\Desktop\rtdetr\aphids_ivan\train\images")
DST = Path(r"C:\Users\sudanb\Desktop\rtdetr\aphids_ivan\train_resized\images")
TARGET_W, TARGET_H = 640, 640   # matches OAK-SR inference resolution

DST.mkdir(parents=True, exist_ok=True)
all_images = sorted(SRC.glob("*.jpg"))
total = len(all_images)

for i, img_path in enumerate(all_images, 1):
    img = Image.open(img_path)
    img.resize((TARGET_W, TARGET_H), Image.BILINEAR).save(DST / img_path.name)
    print(f"[{i}/{total}] {img_path.name}")

print("Done!")