import os
import random
import shutil
from tqdm import tqdm
import argparse

def main(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    existing_files = set(os.listdir(dst_dir))
    all_files = os.listdir(src_dir)
    image_files = [f for f in all_files if f.endswith(".jpg") or f.endswith(".jpeg")]
    random_images = random.sample(image_files, k=400)

    for image in tqdm(random_images):
        src_path = os.path.join(src_dir, image)
        dst_path = os.path.join(dst_dir, image)

        if image in existing_files:
            print(f"Skipping {image} to another dir because it already exists")
        else:
            shutil.copy(src_path, dst_path)
            existing_files.add(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy random image files from source to destination directory.")
    parser.add_argument("--src_dir", required=True, help="Path to the source directory containing image files.")
    parser.add_argument("--dst_dir", required=True, help="Path to the destination directory where image files will be copied.")
    args = parser.parse_args()

    main(args.src_dir, args.dst_dir)
