import os
import random
import shutil
from pathlib import Path

# Paths
base_dir = Path("data/label") #your_dataset
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# Output folders
output_dir = Path("data")
output_base = output_dir / "split"
train_img_dir = output_base / "train" / "images"
train_lbl_dir = output_base / "train" / "labels"
val_img_dir = output_base / "val" / "images"
val_lbl_dir = output_base / "val" / "labels"

# Create output folders
for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Get all image files
image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
random.shuffle(image_files)

# Split ratio
split_ratio = 0.9
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

def copy_files(file_list, img_out, lbl_out):
    for img_file in file_list:
        lbl_file = labels_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy(img_file, img_out / img_file.name)
            shutil.copy(lbl_file, lbl_out / lbl_file.name)

copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print(f"Dataset split complete:")
print(f"→ {len(train_files)} images in train set")
print(f"→ {len(val_files)} images in val set")
