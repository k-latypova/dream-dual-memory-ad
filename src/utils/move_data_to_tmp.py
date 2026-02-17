import os
import shutil
import random
from tqdm import tqdm
from tempfile import TemporaryDirectory
from multiprocessing import Pool, cpu_count

def copy_image(args):
    src_path, dest_path = args
    shutil.copy(src_path, dest_path)

def move_images_parallel(source_dir: str, tmp_base: str, num_images: int, num_cpus: int=4) -> TemporaryDirectory:
    """Moves a specified number of images from the source directory to a temporary directory in parallel.

    Args:
        source_dir (str): The source directory containing images.
        num_images (int): The number of images to move.

    Returns:
        TemporaryDirectory: The temporary directory containing the moved images.
    """

    class_name = os.listdir(source_dir)[0]
    tmp_dir = os.path.join(tmp_base, class_name)
    os.makedirs(tmp_dir)

    images = os.listdir(os.path.join(source_dir, class_name))
    if not images:
        raise ValueError(f"No images found in {source_dir}")

    num_images = min(num_images, len(images))
    selected_images = random.sample(images, num_images)

    # Prepare arguments for parallel processing
    copy_args = [
        (os.path.join(source_dir, class_name, img), os.path.join(tmp_dir, img))
        for img in selected_images
    ]

    # Use a pool of workers to copy files in parallel
    with Pool(processes=num_cpus) as pool:
        list(tqdm(pool.imap(copy_image, copy_args), total=len(copy_args), desc="Copying images"))

    print(f"Moved {num_images} images to {tmp_dir}")

    
