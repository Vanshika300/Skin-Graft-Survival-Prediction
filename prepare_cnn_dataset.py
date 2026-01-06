import os
import shutil
import random

# Paths
SOURCE_DIR = "Data/cnn_raw_images"   # imgs folder you copied
TARGET_DIR = "Data/cnn_images"

TRAIN_RATIO = 0.8

# Create folders
for split in ["train", "test"]:
    for cls in ["infected"]:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

# Get all image files
images = [f for f in os.listdir(SOURCE_DIR)
          if f.lower().endswith((".jpg", ".png", ".jpeg"))]

random.shuffle(images)

split_index = int(len(images) * TRAIN_RATIO)
train_images = images[:split_index]
test_images = images[split_index:]

# Copy files
for img in train_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(TARGET_DIR, "train", "infected", img)
    )

for img in test_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(TARGET_DIR, "test", "infected", img)
    )

print("✅ Dataset prepared successfully!")
print(f"Train images: {len(train_images)}")
print(f"Test images: {len(test_images)}")
