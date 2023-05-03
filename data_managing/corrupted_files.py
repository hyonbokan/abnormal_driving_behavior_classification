import os
from PIL import Image, UnidentifiedImageError
import os
import sys
import json
from PIL import Image
from tqdm import tqdm
root_directory = '/home/dnlab/Data-B/data/train'

for subdir, dirs, files in os.walk(root_directory):
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


