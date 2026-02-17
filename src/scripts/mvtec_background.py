import os
import random
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from rembg import remove
import argparse

INDUSTRIAL_DTD_CLASSES = [
    'porous', 'grid', 'striped', 'woven', 'cracked', 
    'banded', 'fibrous', 'meshed', 'crosshatched', 'knitted',
    'blotchy', 'scaly', 'stained', 'dotted', 'flecked'
]

class DTDTextureLoader:
    """
    Lazy-loading DTD texture manager.
    Only loads textures from disk when they're actually needed.
    """
    
    def __init__(self, dtd_root_dir, texture_classes=None):
        """
        Initialize texture loader without loading any images
        
        Args:
            dtd_root_dir: Path to DTD dataset root (contains 'images' folder)
            texture_classes: List of texture classes to use
        """
        self.dtd_root_dir = Path(dtd_root_dir)
        self.images_dir = self.dtd_root_dir / 'images'
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"DTD images directory not found at {self.images_dir}")
        
        if texture_classes is None:
            texture_classes = INDUSTRIAL_DTD_CLASSES
        
        # Build file index WITHOUT loading images
        self.texture_index = {}
        self._build_index(texture_classes)
        
        if not self.texture_index:
            raise ValueError("No texture files found in DTD directory")
        
        total_files = sum(len(v) for v in self.texture_index.values())
        print(f"DTD texture index loaded: {total_files} files across {len(self.texture_index)} classes")
    
    def _build_index(self, texture_classes):
        """
        Build an index of texture file paths without loading images
        
        Args:
            texture_classes: List of texture classes to index
        """
        for class_name in texture_classes:
            class_dir = self.images_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_name} directory not found")
                continue
            
            # Just collect file paths, don't load images
            file_paths = list(class_dir.glob('*.jpg'))
            
            if file_paths:
                self.texture_index[class_name] = file_paths
    
    def get_random_texture(self):
        """
        Get a random texture image, loading from disk only when called
        
        Returns:
            PIL Image in RGB mode
        """
        # Select random class
        selected_class = random.choice(list(self.texture_index.keys()))
        
        # Select random file from class
        texture_path = random.choice(self.texture_index[selected_class])
        
        # Load only this single image
        try:
            texture = Image.open(texture_path).convert('RGB')
            # Load data into memory (required when using with context manager)
            texture.load()
            return texture
        except Exception as e:
            print(f"Failed to load {texture_path}: {e}")
            # Recursively try again with different texture
            return self.get_random_texture()
    
    def get_texture_by_class(self, class_name):
        """
        Get a random texture from a specific class
        
        Args:
            class_name: Texture class name
        
        Returns:
            PIL Image in RGB mode
        """
        if class_name not in self.texture_index:
            raise ValueError(f"Class {class_name} not found")
        
        texture_path = random.choice(self.texture_index[class_name])
        
        try:
            texture = Image.open(texture_path).convert('RGB')
            texture.load()
            return texture
        except Exception as e:
            print(f"Failed to load {texture_path}: {e}")
            return self.get_texture_by_class(class_name)

def tile_texture(texture_img, target_width, target_height):
    """
    Tile a texture to fit target dimensions
    """
    # Start with texture
    tiled = texture_img.copy()
    
    # Tile horizontally if needed
    while tiled.width < target_width:
        new_tiled = Image.new('RGB', (tiled.width * 2, tiled.height))
        new_tiled.paste(tiled, (0, 0))
        new_tiled.paste(tiled, (tiled.width, 0))
        tiled = new_tiled
    
    # Tile vertically if needed
    while tiled.height < target_height:
        new_tiled = Image.new('RGB', (tiled.width, tiled.height * 2))
        new_tiled.paste(tiled, (0, 0))
        new_tiled.paste(tiled, (0, tiled.height))
        tiled = new_tiled
    
    # Crop to exact size
    tiled = tiled.crop((0, 0, target_width, target_height))
    return tiled

def apply_industrial_processing(texture_img, blur_amount=0):
    """
    Apply subtle processing to make texture look more like industrial background
    """
    if blur_amount > 0:
        texture_img = texture_img.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    
    # Slightly darken to reduce visual prominence
    arr = np.array(texture_img, dtype=np.float32)
    arr = arr * 0.9  # 10% darker
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)

def replace_background_with_dtd(foreground_rgba, texture_loader, blur_background=False):
    """
    Replace background with random DTD texture (loaded on-demand)
    
    Args:
        foreground_rgba: PIL Image in RGBA mode (from rembg)
        texture_loader: DTDTextureLoader instance
        blur_background: Whether to apply subtle blur
    
    Returns:
        PIL Image with new background
    """
    # Load only the texture we need, right now
    texture = texture_loader.get_random_texture()
    
    # Tile texture to match foreground size
    background = tile_texture(texture, foreground_rgba.width, foreground_rgba.height)
    
    # Optional: apply subtle blur
    if blur_background:
        background = apply_industrial_processing(background, blur_amount=2)
    
    # Composite foreground onto texture background
    background.paste(foreground_rgba, (0, 0), foreground_rgba)
    
    # Explicitly delete texture to free memory immediately
    del texture
    
    return background

def process_mvtec_with_dtd(mvtec_input_dir, dtd_root_dir, output_dir, 
                           blur_background=False, keep_original_ratio=0.0):
    """
    Complete pipeline with memory-efficient texture loading
    
    Args:
        mvtec_input_dir: Path to MVTec images
        dtd_root_dir: Path to DTD dataset root
        output_dir: Where to save processed images
        blur_background: Subtle blur on backgrounds
        keep_original_ratio: Ratio of images to keep with original background
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize texture loader (no images loaded yet!)
    print("Initializing DTD texture loader...")
    texture_loader = DTDTextureLoader(dtd_root_dir, INDUSTRIAL_DTD_CLASSES)
    
    # Get MVTec images
    input_path = Path(mvtec_input_dir)
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))

    print(f"Found {len(image_files)} MVTec images...")
    img_indices = np.random.choice(len(image_files), 20, replace=False).tolist()
    image_files = [image_files[idx] for idx in img_indices]
    print(f"Processing {len(image_files)} MVTec images...")
    
    successful = 0
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_file.name}")
        
        # Keep some originals
        if random.random() < keep_original_ratio:
            original = Image.open(img_file)
            original.save(output_path / img_file.name)
            successful += 1
            continue
        
        try:
            # Load MVTec image
            original_img = Image.open(img_file).convert('RGB')
            
            # Remove background
            foreground = remove(np.array(original_img))
            foreground = Image.fromarray(foreground)
            
            # Replace with DTD texture (loaded on-demand)
            result = replace_background_with_dtd(
                foreground, 
                texture_loader, 
                blur_background=blur_background
            )
            
            if result:
                result.save(output_path / img_file.name)
                successful += 1
                
                # Explicitly free memory after each image
                del original_img, foreground, result
        
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    print(f"\nComplete! Successfully processed {successful}/{len(image_files)} images")
    print(f"Output saved to: {output_dir}")

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_path", type=str, required=True, help="Path to MVTEC dataset")
    parser.add_argument("--dtd_path", type=str, required=True, help="Path to DTD dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to store output images")
    args = parser.parse_args()
    process_mvtec_with_dtd(
        mvtec_input_dir=args.mvtec_path,
        dtd_root_dir=args.dtd_path,  # Should contain 'images' subdirectory
        output_dir=args.output_path,
        blur_background=True,
        keep_original_ratio=0.0
    )