import hashlib
import math
import random
import secrets
import cv2
import numpy as np
from sklearn.cluster import KMeans
from perlin_noise import PerlinNoise
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def _sample_random_center(obj_mask_np, blob_w, blob_h, rng, strict_fit=False, max_attempts=1000):
    """
    Sample a random white pixel from mask as blob center.
    
    Args:
        rng: A seeded numpy.random.Generator instance.
        strict_fit: If True, ensures the entire blob rectangle fits inside the mask.
                    If False, only ensures the center point is inside the mask (allows clipping).
    """
    h, w = obj_mask_np.shape
    # Get all valid indices (white pixels)
    y_indices, x_indices = np.where(obj_mask_np > 0)
    
    if len(y_indices) == 0:
        raise ValueError("No object area for blob placement.")

    # If we don't require the full box to fit, we can just pick a random valid pixel.
    # This solves the "always at center" issue by allowing blobs to overlap edges.
    if not strict_fit:
        idx = rng.choice(len(y_indices))
        return y_indices[idx], x_indices[idx]

    # Strict fit logic (original behavior, refined)
    white_pixels = np.column_stack((y_indices, x_indices))
    
    half_h = blob_h // 2
    half_w = blob_w // 2

    for _ in range(max_attempts):
        idx = rng.choice(len(white_pixels))
        center_y, center_x = white_pixels[idx]

        # Quick boundary check first
        y_min, y_max = center_y - half_h, center_y + half_h
        x_min, x_max = center_x - half_w, center_x + half_w
        
        if y_min < 0 or y_max >= h or x_min < 0 or x_max >= w:
            continue

        # Check if region is fully within mask
        # We slice the mask to check the region (faster than loop)
        region = obj_mask_np[y_min:y_max, x_min:x_max]
        if region.size == 0: 
            continue
            
        # If any pixel in the box is 0 (background), it doesn't fit
        if np.any(region == 0):
            continue
            
        return center_y, center_x

    # Fallback if strict fit fails (e.g. blob is too big for complex shape)
    # We return a random valid center rather than crashing
    print(f"Warning: Could not find strict fit after {max_attempts} attempts. Using loose fit.")
    idx = rng.choice(len(y_indices))
    return y_indices[idx], x_indices[idx]

def _perlin_blob(seed=42, size=128):
    # This frequency seems to work fine for blobs
    freq = 333

    # We draw two circular gradient on top of the perlin noise
    # to make sure there is a blob in the center of the image
    max_gradient_dist_1 = size // 2
    max_gradient_dist_2 = size // 4
    gradient_weight_1 = 0.333
    gradient_weight_2 = 0.1666

    # We make the noise a bit more bright to produce larger blobs
    perlin_noise_boost = 0.1666

    noise = PerlinNoise(octaves=4, seed=seed)
    blob_image = Image.new("L", (size, size))

    for x in range(size):
        for y in range(size):
            value = noise([x / freq, y / freq])
            value = value + 0.5 + perlin_noise_boost

            dx = x - size // 2
            dy = y - size // 2
            dist = (dx * dx + dy * dy) ** 0.5
            t1 = dist / max_gradient_dist_1
            t2 = dist / max_gradient_dist_2
            p1 = 1 - t1
            p2 = 1 - t2
            value = value + p1 * gradient_weight_1 + p2 * gradient_weight_2
            value = int(min(max(value * 255, 0), 255))
            blob_image.putpixel((x, y), value)

    contrast = ImageEnhance.Contrast(blob_image)
    blob_image = contrast.enhance(10)

    return blob_image


def _seed_function(seed):
    if seed is None:
        return secrets.randbits(64)
    if isinstance(seed, str):
        hash_object = hashlib.blake2b(
            seed.encode("utf-8"), digest_size=8, salt=b"blob-masks"
        )
        return int.from_bytes(hash_object.digest(), "big")
    if isinstance(seed, (int, float)):
        return int(seed)

    raise ValueError("Invalid seed type. Seed must be a string or a number.")


def _wobbly_effect(
    image,
    amplitude_h,
    frequency_h,
    amplitude_v,
    frequency_v,
    phase_shift_h=0,
    phase_shift_v=0,
):
    width, height = image.size
    new_image = Image.new("RGBA", (width, height))

    for x in range(width):
        for y in range(height):
            dx = int(
                amplitude_h * math.sin(2 * math.pi * y / frequency_h + phase_shift_h)
            )
            dy = int(
                amplitude_v * math.sin(2 * math.pi * x / frequency_v + phase_shift_v)
            )
            new_x = x + dx
            new_y = y + dy

            if 0 <= new_x < width and 0 <= new_y < height:
                new_image.putpixel((x, y), image.getpixel((new_x, new_y)))
            else:
                new_image.putpixel((x, y), (0, 0, 0, 0))

    return new_image


def _wobbly_frame(seed=42, size=128):
    width = size
    height = size
    gradient_size = 10

    image = Image.new("L", (width, height), "black")
    draw = ImageDraw.Draw(image)
    center_rect = (
        gradient_size,
        gradient_size,
        width - gradient_size,
        height - gradient_size,
    )
    for i in range(gradient_size):
        gradient_color = int((255 * i) / gradient_size)
        draw.rectangle(center_rect, fill=gradient_color)
        center_rect = (
            center_rect[0] + 1,
            center_rect[1] + 1,
            center_rect[2] - 1,
            center_rect[3] - 1,
        )

    draw.rectangle((0, 0, width - 1, height - 1), outline="black")

    random.seed(seed * 42)
    ah = random.randint(3, 5)
    fh = random.randint(50, 100)
    av = random.randint(3, 5)
    fv = random.randint(50, 100)
    sh = random.randint(1, 3)
    sv = random.randint(1, 3)
    image = _wobbly_effect(image, ah, fh, av, fv, math.pi / sh, math.pi / sv)

    blur_radius = 5
    image = image.filter(ImageFilter.GaussianBlur(blur_radius))

    return image


def _estimate_bg_color(image, sample_ratio=0.8):
    h, w = image.shape[:2]
    num_samples = int(h * w * sample_ratio)
    indices = np.random.choice(h * w, num_samples, replace=False)
    pixels = image.reshape((-1, 3))[indices]

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    bg_label = np.argmax(np.bincount(labels))
    bg_color = tuple(kmeans.cluster_centers_[bg_label].astype(np.uint8))

    return bg_color


def object_mask_v1(image, bg_color, threshold=30):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bg_lab = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2LAB)[0][0]
    diff = np.linalg.norm(lab.astype(np.float32) - bg_lab.astype(np.float32), axis=2)

    _, mask = cv2.threshold(diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            obj_mask = np.zeros_like(mask)
            cv2.fillPoly(obj_mask, [largest], 255)
            mask = obj_mask

    return mask


def object_mask_v2(image_bgr, bg_color, color_tolerance=30):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2HSV)[0][0]

    lower = np.array(
        [
            max(0, bg_hsv[0] - color_tolerance),
            max(0, bg_hsv[1] - 50),
            max(0, bg_hsv[2] - 50),
        ]
    )
    upper = np.array([min(179, bg_hsv[0] + color_tolerance), 255, 255])

    mask_bg = cv2.inRange(hsv, lower, upper)
    mask_obj = cv2.bitwise_not(mask_bg)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel)
    mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        obj_mask = np.zeros_like(mask_obj)
        cv2.fillPoly(obj_mask, [largest], 255)
        return obj_mask
    else:
        return mask_obj


def _get_object_bbox(mask):
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        raise ValueError("No object detected in mask.")
    y1, x1 = np.min(coords, axis=0)
    y2, x2 = np.max(coords, axis=0)
    return y1, x1, y2, x2



def object_mask_v3(image_bgr, bg_color=None, threshold=15, debug=False):
    """
    More robust object detection for metallic/reflective objects like wallplugs.
    Uses LAB color space (better for perceptual color) + adaptive thresholding.
    """
    if bg_color is None:
        bg_color = _estimate_bg_color(image_bgr, sample_ratio=0.5)
    
    # Convert to LAB (more perceptually uniform than HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    bg_lab = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2LAB)[0][0]
    
    # Compute Euclidean distance in LAB space
    diff = np.linalg.norm(
        lab.astype(np.float32) - bg_lab.astype(np.float32), 
        axis=2
    )
    
    # Adaptive thresholding on distance map
    _, mask = cv2.threshold(diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
    
    # Light morphology (preserve details)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # **Key fix**: Use connected components instead of single largest contour
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # Remove background (label 0) and find largest object
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        if len(areas) > 0 and np.max(areas) > 50:
            largest_label = np.argmax(areas) + 1
            obj_mask = np.zeros_like(mask)
            obj_mask[labels == largest_label] = 255
            return obj_mask
    
    # Fallback: return what we have
    if np.sum(mask) > 100:
        return mask
    
    raise ValueError(f"No object detected (max distance: {np.max(diff):.2f}, threshold: {threshold})")


def object_mask_all_v1(image, bg_color, threshold=30, min_area=50):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bg_lab = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2LAB)[0][0]
    diff = np.linalg.norm(lab.astype(np.float32) - bg_lab.astype(np.float32), axis=2)

    _, mask = cv2.threshold(diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obj_mask = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.fillPoly(obj_mask, [c], 255)

    if np.count_nonzero(obj_mask) == 0:
        raise ValueError("No objects detected")

    return obj_mask

object_mask_funcs = [
    object_mask_v1,
    object_mask_v2,
    object_mask_v3,
    object_mask_all_v1,
]

def object_blob_mask(
    image_path_or_array,
    seed=None,
    threshold=30,
    out_width=512,
    out_height=512,
    blob_ratio=1.0,
    random_position=True,
    use_object_mask=True,
    object_mask_func_version:int = 2,
    bg_color=None,
    strict_fit=False
):
    """
    Generate a blob mask on a new black image, applied only to the object region.

    Args:
        image_path_or_array: Path to image or numpy array (BGR).
        seed: Seed for reproducibility (str, int, or None).
        threshold: Color difference threshold for object detection (higher = stricter).
        out_width: Output image width.
        out_height: Output image height.
        blob_ratio: Relative size of blob to object bounding box (0-1; default 1.0).
        random_position: If True, place blob at random valid position within object.

    Returns:
        PIL Image: Black background with blob on object area (binary), shape (out_width, out_height).
    """
    # Load/convert image to OpenCV format (only needed for sizing when use_object_mask=False,
    # but leaving as-is for simplicity)
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array.copy()
    if image is None:
        raise ValueError("Could not load image.")

    # Resize input to output size (non-square allowed)
    image_resized = cv2.resize(image, (out_width, out_height))

    seed = _seed_function(seed)
    rng = np.random.default_rng(seed)

    # Generate base blob at fixed small square resolution, then resize to target blob size
    base_size = 128
    perlin_blob_image = _perlin_blob(seed, size=base_size).convert("L")
    wobbly_frame_image = _wobbly_frame(seed, size=base_size).convert("L")

    mix_small = Image.new("L", (base_size, base_size))
    for x in range(base_size):
        for y in range(base_size):
            perlin_val = perlin_blob_image.getpixel((x, y))
            frame_val = wobbly_frame_image.getpixel((x, y))
            pixel = int(perlin_val * frame_val / 255)
            mix_small.putpixel((x, y), pixel)

    contrast = ImageEnhance.Contrast(mix_small)
    mix_small = contrast.enhance(10)
    object_mask_func = object_mask_funcs[object_mask_func_version - 1]

    # If using object mask, compute blob size from object bbox.
    # Otherwise, use blob_ratio relative to min(out_width, out_height).
    if use_object_mask:
        # Estimate background color
        if not bg_color:
            bg_color = _estimate_bg_color(image_resized)
        else:
            bg_color = tuple(bg_color)

        # Create object mask
        obj_mask = object_mask_func(image_resized, bg_color, threshold)
        obj_mask_np = obj_mask.astype(np.uint8)

        # Get object bounding box
        y1, x1, y2, x2 = _get_object_bbox(obj_mask)
        obj_w = max(1, x2 - x1)
        obj_h = max(1, y2 - y1)

        blob_w = int(obj_w * blob_ratio)
        blob_h = int(obj_h * blob_ratio)
    else:
        # Blob size relative to image, no object
        base_dim = min(out_width, out_height)
        blob_w = int(base_dim * blob_ratio)
        blob_h = int(base_dim * blob_ratio)

        blob_w = max(1, min(blob_w, out_width))
        blob_h = max(1, min(blob_h, out_height))

    blob_w = max(1, blob_w)
    blob_h = max(1, blob_h)

    # Resize to rectangular blob size
    blob_resized = mix_small.resize((blob_w, blob_h), Image.LANCZOS)

    def threshold_fn(img):
        return img.point(lambda x: 0 if x == 0 else 255, mode="1")

    blob_binary = threshold_fn(blob_resized)

    # Determine paste position
    if use_object_mask:
        if random_position:
            # PASS RNG and strict_fit to the function
            center_y, center_x = _sample_random_center(
                obj_mask_np, blob_w, blob_h, rng=rng, strict_fit=strict_fit
            )
            paste_x = center_x - blob_w // 2
            paste_y = center_y - blob_h // 2
        else:
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            paste_x = obj_center_x - blob_w // 2
            paste_y = obj_center_y - blob_h // 2
    else:
        if random_position:
            max_x = out_width - blob_w
            max_y = out_height - blob_h
            # Use RNG here as well
            paste_x = rng.integers(0, max_x + 1) if max_x > 0 else 0
            paste_y = rng.integers(0, max_y + 1) if max_y > 0 else 0
        else:
            paste_x = (out_width - blob_w) // 2
            paste_y = (out_height - blob_h) // 2

    # Clamp paste position to image bounds
    paste_x = max(0, min(paste_x, out_width - blob_w))
    paste_y = max(0, min(paste_y, out_height - blob_h))

    # Output image
    output = Image.new("1", (out_width, out_height), 0)

    if use_object_mask:
        # Clip with object mask to ensure no overflow outside object
        obj_mask_pil = Image.fromarray(obj_mask).convert("1")
        final = Image.new("1", (out_width, out_height), 0)
        temp_pasted = Image.new("1", (out_width, out_height), 0)
        temp_pasted.paste(blob_binary, (paste_x, paste_y), blob_binary)
        final.paste(temp_pasted, (0, 0), obj_mask_pil)
        return final
    else:
        # No object mask, just paste blob directly
        output.paste(blob_binary, (paste_x, paste_y), blob_binary)
        return output