import os
from PIL import Image


if __name__ == "__main__":
    root_dir = "/scratch/latypova/data/mvtec"
    classes = os.listdir(root_dir)
    classes = [d for d in classes if os.path.isdir(os.path.join(root_dir, d))]
    resolutions = {}
    for cl in classes:
        imgname = list(os.listdir(os.path.join(root_dir, cl, "train", "good")))[0]
        img = Image.open(os.path.join(root_dir, cl, "train", "good", imgname))
        resolutions[cl] = img.size
    print(resolutions)