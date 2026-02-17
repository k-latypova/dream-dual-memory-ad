import math
import cv2
import numpy as np
from skimage.exposure import adjust_gamma
from PIL import Image


# numpy arrays from PIL are RGB; use RGB conversions
def reinhard_color_transfer(src_rgb, tgt_rgb) -> np.array:
    src = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(tgt_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    for i in range(3):
        s_mean, s_std = src[..., i].mean(), src[..., i].std()
        t_mean, t_std = tgt[..., i].mean(), tgt[..., i].std()
        # avoid divide-by-zero
        if t_std < 1e-6:
            t_std = 1e-6
        tgt[..., i] = (tgt[..., i] - t_mean) * (s_std / t_std) + s_mean 

    tgt = np.clip(tgt, 0, 255).astype(np.uint8)
    return cv2.cvtColor(tgt, cv2.COLOR_LAB2RGB)


def adjust_exposure(image: np.array, gamma=0.8) -> np.array:
    result = (adjust_gamma(image.astype("float32")/255.0, gamma=gamma)*255).astype("uint8")
    return result

def put_watermark(img, wm_encoder):
    img = cv2.cvtColor(np.array(img, cv2.COLOR_RGB2BGR))
    img = wm_encoder.encode(img, 'dwtDCT')
    img = Image.fromarray(img[:, :, ::-1])
    return img

def crop_mask_with_padding(image: Image, mask: Image, ratio: float,
                           target_img_size):
    mask_np = np.array(mask)
    coords = np.column_stack(np.where(mask_np > 128))
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) 
    h, w = mask_np.shape
    mask_h = (y2-y1)
    mask_w = (x2-x1)
    print(f"mask h: {mask_h}, mask_w: {mask_w}")
    a = 4
    b = 2 * (mask_h + mask_w)
    c = mask_h * mask_w - (mask_np.sum() / (255 *ratio))
    D = b**2 - 4*a*c
    sqrt_D = math.sqrt(D)
    p = (-b + sqrt_D) / (2*a)
    padding = int(p)

    y1 = max(0, y1 - padding)
    x1 = max(0, x1 - padding)
    y2 = min(h, y2 + padding)
    x2 = min(w, x2 + padding)
    crop_box = (x1, y1, x2, y2)
    cropped_image = image.crop(crop_box)
    cropped_mask = mask.crop(crop_box)
    print(f"Calculated padding: {p}, current_ratio: {(np.array(cropped_mask).astype(np.float32) / 255.0).mean()}")

    cropped_image = resize_pil_img_with_cv2(cropped_image, target_img_size)
    cropped_mask = cropped_mask.resize(target_img_size)

    return cropped_image, cropped_mask, crop_box


def resize_pil_img_with_cv2(pil_img: Image, target_size) -> Image:
    img_np = np.array(pil_img)
    h, w = img_np.shape[:2]
    if (h, w) != target_size:
        img_np = cv2.resize(img_np, target_size[::-1], interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_np)