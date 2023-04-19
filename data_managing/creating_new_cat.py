import os
import random
import shutil
from tqdm import tqdm

src_dir = '/media/Data-B/data/main_data/val_ppr_cat/driving'
dst_dir = '/media/Data-B/data/main_data/val_cat_new/drving'

# Create the destination directory if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Get a list of all the files in the destination directory
existing_files = set(os.listdir(dst_dir))

# Get a list of all the image files in the source directory
all_files = os.listdir(src_dir)
image_files = [f for f in all_files if f.endswith(".jpg") or f.endswith(".jpeg")]

# Get a random sample of 10,000 image files
random_images = random.sample(image_files, k=400)

# Copy the random images to the destination directory
for image in tqdm(random_images):
    src_path = os.path.join(src_dir, image)
    dst_path = os.path.join(dst_dir, image)

    if image in existing_files:
        print(f"skipping {image} to another dir because it already exists")
        
    else:
        shutil.copy(src_path, dst_path)
        existing_files.add(image)


