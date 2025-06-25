import os
import cv2
import albumentations as A

# Input and output directories
input_dir = r'C:\Users\Sujan.S\OneDrive\Documents\all data\bottle\tupperware'
output_dir = r'C:\Users\Sujan.S\OneDrive\Documents\all data\bottle\tupperware_aug'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.RandomShadow(p=0.3),
    A.RandomRain(p=0.2),
    A.RandomSunFlare(p=0.2)
])

# Loop through images
for img_file in os.listdir(input_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] Failed to load image: {img_path}")
            continue

        for i in range(14):  # Generate 10 augmentations
            augmented = transform(image=image)['image']
            new_filename = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
            output_path = os.path.join(output_dir, new_filename)
            cv2.imwrite(output_path, augmented)

print("✅ Augmentation completed!")







