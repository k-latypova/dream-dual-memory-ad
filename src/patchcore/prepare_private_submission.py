import os
import numpy as np
import PIL
from tifffile import imwrite
from tqdm.auto import tqdm
import scipy.misc

def save_image_segmentation_submission(
    savefolder,
    classname,
    image_paths,
    segmentations,
    segmentations_thresholded=None,
):
    """Generate anomaly segmentation images for private submission.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        image_transform: [function or lambda] Optional transformation of images.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    os.makedirs(savefolder, exist_ok=True)

    non_binary_path = os.path.join(savefolder, "anomaly_images", classname)
    binary_path = os.path.join(savefolder, "anomaly_images_thresholded", classname)

    # Non thresholded segmentation images
    os.makedirs(os.path.join(non_binary_path, "test_private"), exist_ok=True)
    os.makedirs(os.path.join(non_binary_path, "test_private_mixed"), exist_ok=True)
    os.makedirs(os.path.join(binary_path, "test_private"), exist_ok=True)
    os.makedirs(os.path.join(binary_path, "test_private_mixed"), exist_ok=True)
    print(f"{len(image_paths)} segmentation images to be saved to {non_binary_path} and {binary_path}.")


    for image_path, segmentation, segmentation_thresholded in tqdm(
        zip(image_paths, segmentations, segmentations_thresholded),
        total=len(image_paths),
        desc="Generating Segmentation Images for Submission...",
        leave=False,
    ):
        # Normalize segmentation to [0, 255]
        seg_min, seg_max = segmentation.min(), segmentation.max()
        #segmentation_thresholded = (segmentation > threshold).astype(np.uint8)
        # if seg_max - seg_min > 0:
        #     segmentation = (segmentation - seg_min) / (seg_max - seg_min)
        # else:
        #     segmentation = segmentation * 0.0
        #segmentation_thresholded = (segmentation > threshold).astype(np.uint8)
        #imagename = os.path.basename(image_path)
        segmentation = segmentation.astype(np.float16) * 255.0
        segmentation_thresholded = (segmentation_thresholded * 255).astype(np.uint8)

        
        savename, filename = image_path.split("/")[-2], image_path.split("/")[-1]
        non_bin_filename = filename.split(".")[0] + ".tiff"
        binary_filename = filename.split(".")[0] + ".png"        

        non_binary_savename = os.path.join(non_binary_path, savename, non_bin_filename)
        binary_savename = os.path.join(binary_path, savename, binary_filename)

        # anomaly_mask_img = PIL.Image.fromarray(segmentation).convert('L')
        # anomaly_mask_img.save(non_binary_savename, format='TIFF')
        imwrite(non_binary_savename, segmentation)
        anomaly_mask_thresh_img = PIL.Image.fromarray(segmentation_thresholded).convert('L')
        anomaly_mask_thresh_img.save(binary_savename, format='PNG')


