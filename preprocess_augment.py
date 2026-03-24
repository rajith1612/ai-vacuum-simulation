import os
import random
import shutil
from PIL import Image, ImageEnhance
from tqdm import tqdm

# Configuration
input_root = 'dataset'              # Your original dataset folder
output_root = 'processed_dataset'   # New resized and augmented dataset folder
target_size = (64, 64)               # Resize to 64x64

# Augmentation function for dusty images
def augment_image(image):
    augmentations = []

    # Rotate
    angle = random.choice([10, -10, 15, -15])
    rotated = image.rotate(angle)
    augmentations.append(rotated)

    # Flip horizontally
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmentations.append(flipped)

    # Brightness change
    enhancer = ImageEnhance.Brightness(image)
    brightness = enhancer.enhance(random.uniform(0.7, 1.3))
    augmentations.append(brightness)

    # Zoom-in
    zoomed = image.crop((10, 10, 54, 54)).resize(target_size)
    augmentations.append(zoomed)

    return augmentations

# Walk through all folders
for root, dirs, files in os.walk(input_root):
    for file in tqdm(files):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            
            # Open and resize image
            img = Image.open(file_path).convert('RGB')
            img = img.resize(target_size)

            # Prepare output path
            relative_path = os.path.relpath(root, input_root)
            output_folder = os.path.join(output_root, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            # Save resized image
            base_name = os.path.splitext(file)[0]
            img.save(os.path.join(output_folder, f"{base_name}.jpg"))

            # Apply augmentations if it's dusty class
            if 'dusty_wooden' in root or 'dusty_marble' in root:
                augmentations = augment_image(img)
                for idx, aug_img in enumerate(augmentations):
                    aug_img.save(os.path.join(output_folder, f"{base_name}_aug{idx}.jpg"))

print("✅ Preprocessing and augmentation completed!")