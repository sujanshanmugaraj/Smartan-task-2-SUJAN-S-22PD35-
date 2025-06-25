# import os
# import shutil
# import random

# # Set seed
# random.seed(42)

# # Source root where all category folders are stored
# source_root = r"C:\Users\Sujan.S\OneDrive\Documents\all data"
# target_root = r"C:\Users\Sujan.S\OneDrive\Documents\SmartanAI-YOLO-Custom-Detection\dataset"

# # Subfolders to walk through
# categories = {
#     'bottle': ['borosil', 'plastic', 'tupperware'],
#     'flowers': ['daisy', 'hibiscus', 'rose'],
#     'tools': ['hammer', 'skrewdriver', 'spanner']
# }

# # Create target directories
# for split in ['train', 'val']:
#     os.makedirs(os.path.join(target_root, 'images', split), exist_ok=True)
#     os.makedirs(os.path.join(target_root, 'labels', split), exist_ok=True)

# all_files = []

# # Collect all image-label pairs with class name as prefix
# for main_cat, subcats in categories.items():
#     for subcat in subcats:
#         folder = os.path.join(source_root, main_cat, subcat)
#         for file in os.listdir(folder):
#             if file.endswith('.jpg'):
#                 base = os.path.splitext(file)[0]
#                 label_file = base + '.txt'
#                 img_path = os.path.join(folder, file)
#                 label_path = os.path.join(folder, label_file)

#                 if os.path.exists(label_path):
#                     new_base_name = f"{subcat}_{base}"
#                     all_files.append((img_path, label_path, new_base_name))
#                 else:
#                     print(f"⚠️ Missing label for {file}")

# # Shuffle and split
# random.shuffle(all_files)
# split_idx = int(0.8 * len(all_files))
# train_set = all_files[:split_idx]
# val_set = all_files[split_idx:]

# # Copy function
# def copy_files(file_list, split):
#     for img_path, label_path, new_base in file_list:
#         dst_img = os.path.join(target_root, 'images', split, new_base + '.jpg')
#         dst_lbl = os.path.join(target_root, 'labels', split, new_base + '.txt')
#         shutil.copy2(img_path, dst_img)
#         shutil.copy2(label_path, dst_lbl)

# # Copy into YOLO folders
# copy_files(train_set, 'train')
# copy_files(val_set, 'val')

# print("✅ Dataset organized and split into train/val successfully.")



import os
import re

# Base folder of YOLOv5 (edit if needed)
YOLOV5_DIR = "yolov5"

def patch_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    changed = False

    for line in lines:
        # Convert 'from yolov5.utils.general import ...' -> 'from .utils.general import ...'
        if "from yolov5." in line:
            line = re.sub(r"from yolov5\.", "from .", line)
            changed = True
        elif "import yolov5." in line:
            line = re.sub(r"import yolov5\.", "import .", line)
            changed = True

        updated_lines.append(line)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
        print(f"✅ Patched: {filepath}")
    else:
        print(f"⏭️ Skipped: {filepath}")

def patch_yolov5():
    for root, _, files in os.walk(YOLOV5_DIR):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                patch_file(full_path)

if __name__ == "__main__":
    print("🔧 Starting patch for YOLOv5 import paths...")
    patch_yolov5()
    print("✅ All applicable files patched.")

