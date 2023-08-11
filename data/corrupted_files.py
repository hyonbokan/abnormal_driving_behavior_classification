import os
from PIL import Image
from tqdm import tqdm
import argparse

def main(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for filename in tqdm(files):
            if filename.endswith(".jpg"):
                # try to open the file as an image
                try:
                    img = Image.open(os.path.join(subdir, filename))
                    img.verify()
                except:
                    # if the file is truncated or corrupted, delete it
                    os.remove(os.path.join(subdir, filename))
                    print(f"Deleted truncated file: {os.path.join(subdir, filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete truncated or corrupted image files in a directory.")
    parser.add_argument("--root_dir", required=True, help="Path to the root directory containing image files.")
    args = parser.parse_args()

    main(args.root_directory)
