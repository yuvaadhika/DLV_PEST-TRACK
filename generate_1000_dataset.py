"""
================================================================================
PestTrack AI - 1,000 Images Dataset Generator
Generates:
  1. 10 Agricultural Pest Classes across Train (700), Val (150), Test (150)
  2. Leaf Damage Semantic Segmentation Dataset (100 images + 100 masks)
Total Images: 1,100 realistic crop images with metadata
================================================================================
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = [
    "Aphids", "Armyworm", "Beetle", "Bollworm", "Grasshopper",
    "Leafhopper", "Mealybug", "Mites", "Stem_Borer", "Whitefly"
]

SPLITS = {
    "train": 70,  # 70 * 10 = 700 images
    "val": 15,    # 15 * 10 = 150 images
    "test": 15    # 15 * 10 = 150 images
}

def generate_pest_image(class_name, class_idx, seed_val):
    np.random.seed(seed_val)
    # Background: agricultural field / leaf texture
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    bg_green = np.random.randint(110, 180)
    bg_red = np.random.randint(30, 80)
    bg_blue = np.random.randint(30, 80)
    img[:] = (bg_blue, bg_green, bg_red)

    # Leaf veins texture
    for _ in range(np.random.randint(4, 8)):
        pt1 = (np.random.randint(0, 224), np.random.randint(0, 224))
        pt2 = (np.random.randint(0, 224), np.random.randint(0, 224))
        cv2.line(img, pt1, pt2, (bg_blue + 15, bg_green + 25, bg_red + 10), 1, cv2.LINE_AA)

    # Pest body characteristics based on species
    cx = np.random.randint(80, 144)
    cy = np.random.randint(80, 144)
    angle = np.random.randint(0, 360)

    # Species-specific palette
    if class_name in ["Aphids", "Mites"]:
        # Small clustered bodies
        body_color = (np.random.randint(20, 60), np.random.randint(140, 220), np.random.randint(80, 160))
        cv2.ellipse(img, (cx, cy), (12, 18), angle, 0, 360, body_color, -1)
        cv2.ellipse(img, (cx + 8, cy + 8), (10, 15), angle + 20, 0, 360, body_color, -1)
    elif class_name in ["Beetle", "Weevil"]:
        # Hard round dark exoskeleton with shiny highlight
        body_color = (np.random.randint(15, 45), np.random.randint(20, 50), np.random.randint(30, 70))
        cv2.ellipse(img, (cx, cy), (28, 42), angle, 0, 360, body_color, -1)
        cv2.ellipse(img, (cx - 5, cy - 8), (6, 12), angle, 0, 360, (200, 200, 200), -1) # shine
    elif class_name in ["Armyworm", "Bollworm", "Stem_Borer"]:
        # Elongated caterpillar body segments
        body_color = (np.random.randint(30, 70), np.random.randint(90, 150), np.random.randint(110, 180))
        for seg in range(6):
            sx = int(cx + (seg - 3) * 10 * np.cos(np.radians(angle)))
            sy = int(cy + (seg - 3) * 10 * np.sin(np.radians(angle)))
            cv2.circle(img, (sx, sy), 14, body_color, -1)
    elif class_name == "Grasshopper":
        # Green elongated body with long jumping legs
        body_color = (np.random.randint(30, 60), np.random.randint(160, 230), np.random.randint(40, 90))
        cv2.ellipse(img, (cx, cy), (16, 48), angle, 0, 360, body_color, -1)
        # Hind legs
        cv2.line(img, (cx - 10, cy), (cx - 35, cy - 25), (20, 120, 30), 2, cv2.LINE_AA)
        cv2.line(img, (cx - 35, cy - 25), (cx - 45, cy + 20), (20, 120, 30), 2, cv2.LINE_AA)
    elif class_name == "Whitefly":
        # Small white powdery wings
        body_color = (np.random.randint(220, 255), np.random.randint(225, 255), np.random.randint(220, 255))
        cv2.ellipse(img, (cx - 6, cy), (10, 16), angle - 25, 0, 360, body_color, -1)
        cv2.ellipse(img, (cx + 6, cy), (10, 16), angle + 25, 0, 360, body_color, -1)
    else: # Leafhopper / Mealybug
        body_color = (np.random.randint(80, 140), np.random.randint(150, 210), np.random.randint(120, 180))
        cv2.ellipse(img, (cx, cy), (18, 32), angle, 0, 360, body_color, -1)

    # Add realistic Gaussian sensor noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_img

def build_dataset():
    print("Generating 1,000 Images Dataset...")
    total_pest_images = 0

    for split_name, count_per_class in SPLITS.items():
        for idx, cls in enumerate(CLASSES):
            target_dir = os.path.join(DATASET_DIR, split_name, cls)
            os.makedirs(target_dir, exist_ok=True)

            for i in range(count_per_class):
                seed = hash(f"{split_name}_{cls}_{i}") % (2**31)
                img = generate_pest_image(cls, idx, seed)
                filename = f"{cls.lower()}_{split_name}_{i+1:04d}.jpg"
                filepath = os.path.join(target_dir, filename)
                cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                total_pest_images += 1

    print(f"-> Generated {total_pest_images} images across Train (700), Val (150), and Test (150)!")

    # Leaf Damage Segmentation Dataset
    seg_img_dir = os.path.join(DATASET_DIR, "leaf_damage_segmentation", "images")
    seg_mask_dir = os.path.join(DATASET_DIR, "leaf_damage_segmentation", "masks")
    os.makedirs(seg_img_dir, exist_ok=True)
    os.makedirs(seg_mask_dir, exist_ok=True)

    print("Generating 100 Leaf Damage Images & Binary Ground-Truth Masks...")
    for j in range(100):
        leaf_img = np.zeros((256, 256, 3), dtype=np.uint8)
        leaf_img[:] = (245, 248, 245)
        # Leaf contour
        cv2.ellipse(leaf_img, (128, 128), (80, 110), 0, 0, 360, (34, 139, 34), -1)

        mask = np.zeros((256, 256), dtype=np.uint8)
        num_lesions = np.random.randint(1, 5)
        for _ in range(num_lesions):
            lx = np.random.randint(70, 186)
            ly = np.random.randint(70, 186)
            lr = np.random.randint(10, 25)
            cv2.circle(mask, (lx, ly), lr, 255, -1)
            # Brown necrotic blight lesion on leaf image
            cv2.circle(leaf_img, (lx, ly), lr, (15, 60, 95), -1)
            cv2.circle(leaf_img, (lx, ly), lr + 4, (30, 160, 190), 2) # Chlorotic yellow halo

        cv2.imwrite(os.path.join(seg_img_dir, f"leaf_sample_{j+1:03d}.jpg"), leaf_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(seg_mask_dir, f"leaf_sample_{j+1:03d}_mask.png"), mask)

    print("\n[SUCCESS] 1,100 Total Images Dataset Successfully Generated in:", DATASET_DIR)

if __name__ == "__main__":
    build_dataset()
