from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image, ImageFilter, ExifTags
from PIL.ImageOps import exif_transpose
import os
import numpy as np
from pathlib import Path
from rembg import remove
from numpy import random

CATEGORIES_NO_BG_REPLACEMENT = ['carpet', 'leather', 'grid', 'wood', 'tile', 'zipper', "rice", "fabric", "sheet_metal", "transistor"]  # Extend as needed



def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

MVTEC_TO_DTD_MAPPING = {
    'carpet': ['grid', 'woven', 'striped', 'crosshatched'],
    'grid': ['porous', 'meshed', 'dotted', 'blotchy'],
    'leather': ['cracked', 'scaly', 'stained', 'blotchy'],
    'tile': ['porous', 'dotted', 'cracked', 'banded'],
    'wood': ['banded', 'striped', 'fibrous', 'cracked'],
    'bottle': ['grid', 'striped', 'dotted', 'crosshatched'],
    'cable': ['meshed', 'woven', 'grid', 'porous'],
    'capsule': ['dotted', 'grid', 'striped', 'banded'],
    'hazelnut': ['porous', 'cracked', 'dotted', 'grid'],
    'metal_nut': ['grid', 'crosshatched', 'striped', 'dotted'],
    'pill': ['grid', 'dotted', 'striped', 'banded'],
    'screw': ['grid', 'crosshatched', 'pleated', 'perforated'],
    'toothbrush': ['woven', 'fibrous', 'crosshatched', 'striped'],
    'transistor': ['grid', 'dotted', 'crosshatched', 'banded'],
    'zipper': ['striped', 'woven', 'crosshatched', 'grid'],
}

DTD_GOOD_BACKGROUNDS = {
     'perforated': ['perforated_0013.jpg', 'perforated_0024.jpg', 'perforated_0026.jpg', 'perforated_0053.jpg', 'perforated_0069.jpg', 'perforated_0088.jpg', 'perforated_0118.jpg', 'perforated_0136.jpg', 'perforated_0135.jpg', 'perforated_0140.jpg', 'perforated_0154.jpg', 'perforated_0156.jpg'],
     'crosshatched': ['crosshatched_0033.jpg', 'crosshatched_0044.jpg', 'crosshatched_0050.jpg', 'crosshatched_0053.jpg', 'crosshatched_0058.jpg', 'crosshatched_0078.jpg', 'crosshatched_0099.jpg', 'crosshatched_0168.jpg'],
     'grid': ['grid_0016.jpg', 'grid_0021.jpg', 'grid_0036.jpg', 'grid_0041.jpg', 'grid_0067.jpg', 'grid_0072.jpg', 'grid_0077.jpg'],
     'pleated': ['pleated_0054.jpg', 'pleated_0066.jpg', 'pleated_0067.jpg', 'pleated_0068.jpg', 'pleated_0095.jpg'],
     'cracked': ['cracked_0059.jpg', 'cracked_0059.jpg', 'cracked_0065.jpg', 'cracked_0078.jpg', 'cracked_0079.jpg', 'cracked_0091.jpg', 'cracked_0110.jpg', 'cracked_0111.jpg', 'cracked_0118.jpg', 'cracked_0152.jpg', 'cracked_0158.jpg', 'cracked_0132.jpg'],
     'marbled': ['marbled_0075.jpg', 'marbled_0076.jpg', 'marbled_0097.jpg', 'marbled_0106.jpg', 'marbled_0107.jpg', 'marbled_0109.jpg', 'marbled_0110.jpg', 'marbled_0112.jpg', 'marbled_0114.jpg', 'marbled_0115.jpg', 'marbled_0119.jpg', 'marbled_0125.jpg'],
     'flacked': ['flacked_0048.jpg', 'flacked_0052.jpg', 'flacked_0060.jpg', 'flacked_0067.jpg', 'flacked_0063.jpg', 'flacked_0071.jpg', 'flacked_0075.jpg', 'flacked_0088.jpg', 'flacked_0076.jpg', 'flacked_0094.jpg', 'flacked_0115.jpg']
}


class DTDTextureLoader:
    """
    Lazy-loading DTD texture manager.
    Only loads textures from disk when they're actually needed.
    Minimizes memory footprint while providing diverse backgrounds.
    """
    
    def __init__(self, dtd_root_dir):
        """
        Initialize texture loader without loading any images
        
        Args:
            dtd_root_dir: Path to DTD dataset root (contains 'images' folder)
        """
        self.dtd_root_dir = Path(dtd_root_dir)
        self.images_dir = self.dtd_root_dir / 'images'
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"DTD images directory not found at {self.images_dir}")
        
        # Build file index WITHOUT loading images
        self.texture_index = {}
        self._build_index()
        
        if not self.texture_index:
            raise ValueError("No texture files found in DTD directory")
        
        total_files = sum(len(v) for v in self.texture_index.values())
        print(f"[DTD Loader] Indexed {total_files} files across {len(self.texture_index)} texture classes")
    
    def _build_index(self):
        """Build an index of texture file paths without loading images"""
        for class_dir in self.images_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            filenames = DTD_GOOD_BACKGROUNDS.get(class_name, [])
            file_paths = list(class_dir.glob('*.jpg'))
            if len(filenames) > 0:
                file_paths = [os.path.join(class_dir, x) for x in filenames]
            else:
                file_paths = []
            
            if file_paths:
                self.texture_index[class_name] = file_paths
    
    def get_random_texture_for_class(self, texture_classes):
        """
        Get a random texture from specified classes, loading from disk only when called
        
        Args:
            texture_classes: List of texture class names to choose from
        
        Returns:
            PIL Image in RGB mode
        """
        # Filter to classes that exist
        available_classes = [c for c in texture_classes if c in self.texture_index]
        
        if not available_classes:
            print(f"Warning: No available texture classes from {texture_classes}")
            print(f"Available classes: {list(self.texture_index.keys())}")
            # Fall back to any available class
            available_classes = list(self.texture_index.keys())
        
        selected_class = random.choice(available_classes)
        texture_path = random.choice(self.texture_index[selected_class])
        
        try:
            texture = Image.open(texture_path).convert('RGB')
            texture.load()  # Force load into memory
            return texture
        except Exception as e:
            print(f"Failed to load {texture_path}: {e}")
            return self.get_random_texture_for_class(texture_classes)


def exif_transpose(img):
    """
    Apply exif transpose if it exists, else return image as-is
    """
    if not img:
        return img
    
    try:
        exif = img._getexif()
        if exif is not None:
            exif = dict(exif.items())
            orientation = exif.get(274)  # 274 is the EXIF orientation tag
            if orientation == 3:
                return img.rotate(180, expand=True)
            elif orientation == 6:
                return img.rotate(270, expand=True)
            elif orientation == 8:
                return img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    
    return img


def tile_texture(texture_img, target_width, target_height):
    """Tile a texture to fit target dimensions"""
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
    return tiled.crop((0, 0, target_width, target_height))


def apply_industrial_processing(texture_img, blur_amount=2):
    """Apply subtle processing to make texture look more like industrial background"""
    # Subtle blur to reduce visual prominence
    if blur_amount > 0:
        texture_img = texture_img.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    
    # Slightly darken to reduce visual prominence
    arr = np.array(texture_img, dtype=np.float32)
    arr = arr * 0.9  # 10% darker
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def replace_background_with_dtd(foreground_rgba, texture_loader, texture_classes, blur_background=True):
    """
    Replace background with random DTD texture (loaded on-demand)
    
    Args:
        foreground_rgba: PIL Image in RGBA mode (from rembg)
        texture_loader: DTDTextureLoader instance
        texture_classes: List of texture classes appropriate for this category
        blur_background: Whether to apply subtle blur
    
    Returns:
        PIL Image with new background in RGB mode
    """
    # Load only the texture we need, right now
    texture = texture_loader.get_random_texture_for_class(texture_classes)
    
    # Tile texture to match foreground size
    background = tile_texture(texture, foreground_rgba.width, foreground_rgba.height)
    
    # Apply subtle blur
    if blur_background:
        background = apply_industrial_processing(background, blur_amount=2)
    
    # Composite foreground onto texture background
    background.paste(foreground_rgba, (0, 0), foreground_rgba)
    
    # Explicitly delete texture to free memory immediately
    del texture
    
    return background

class DreamBoothDataset(Dataset):
    """
    DreamBooth dataset with category-aware DTD background augmentation.
    
    Features:
    - On-demand DTD texture loading (memory efficient)
    - Category-specific texture selection for MVTec objects
    - Background removal via rembg
    - Lazy loading of both instance and class images
    """
    
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        crop_size=512,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        instance_data_files=None,
        # New parameters for DTD background augmentation
        dtd_root_dir=None,
        category=None,
        enable_background_augmentation=True,
        background_augmentation_probability=1.0,
    ):
        """
        Initialize DreamBooth dataset with optional DTD background augmentation
        
        Args:
            instance_data_root: Path to instance images
            instance_prompt: Text prompt for instances
            tokenizer: Tokenizer for text encoding
            class_data_root: Path to class images (optional)
            class_prompt: Text prompt for class images
            class_num: Max number of class images to use
            size: Image size
            center_crop: Whether to center crop or random crop
            crop_size: Crop size
            encoder_hidden_states: Pre-encoded instance prompts
            class_prompt_encoder_hidden_states: Pre-encoded class prompts
            tokenizer_max_length: Max length for tokenizer
            instance_data_files: Specific instance files to use
            dtd_root_dir: Path to DTD dataset root (required for background augmentation)
            mvtec_category: MVTec category name (e.g., 'hazelnut', 'bottle')
            enable_background_augmentation: Whether to apply DTD background replacement
            background_augmentation_probability: Probability of applying background augmentation (0-1)
        """
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        
        # Background augmentation parameters
        self.enable_background_augmentation = enable_background_augmentation
        self.background_augmentation_probability = background_augmentation_probability
        self.dtd_texture_loader = None
        self.dtd_texture_classes = None
        self.category = category
        
        # Initialize DTD texture loader if enabled
        if enable_background_augmentation and dtd_root_dir:
            try:
                self.dtd_texture_loader = DTDTextureLoader(dtd_root_dir)
                
                # Get texture classes for this MVTec category
                if category:
                    category = category.lower().strip()
                    if category in MVTEC_TO_DTD_MAPPING:
                        self.dtd_texture_classes = MVTEC_TO_DTD_MAPPING[category]
                        print(f"[DreamBooth] Using DTD textures for '{category}': {self.dtd_texture_classes}")
                    else:
                        print(f"[DreamBooth] Warning: '{category}' not in mapping. Using random DTD textures.")
                        self.dtd_texture_classes = None
                else:
                    print("[DreamBooth] No MVTec category specified. Using random DTD textures.")
                    self.dtd_texture_classes = None
            except Exception as e:
                print(f"[DreamBooth] Failed to load DTD textures: {e}")
                print("[DreamBooth] Proceeding without background augmentation")
                self.enable_background_augmentation = False
        
        # Instance images
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exist.")
        
        self.instance_images_path = (
            list(Path(instance_data_root).iterdir()) 
            if not instance_data_files 
            else [os.path.join(instance_data_root, x) for x in instance_data_files]
        )
        self.num_instance_images = len(self.instance_images_path) if not instance_data_files else len(instance_data_files)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        
        # Class images
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        
        # Image transforms
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(crop_size) if center_crop else transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return self._length
    
    def _apply_dtd_background_augmentation(self, image):
        """
        Apply DTD background augmentation to an image
        
        Args:
            image: PIL Image in RGB mode
        
        Returns:
            PIL Image with potentially augmented background
        """
        if not self.enable_background_augmentation:
            return image
        
        if random.random() > self.background_augmentation_probability:
            return image
        
        if not self.dtd_texture_loader:
            return image

        if self.category in CATEGORIES_NO_BG_REPLACEMENT:
            return image
        
        try:
            # Remove background
            image_array = np.array(image)
            foreground_rgba = remove(image_array)
            foreground_rgba = Image.fromarray(foreground_rgba)
            
            # Replace with DTD texture
            # If no specific categories, pick random from loader
            if self.dtd_texture_classes:
                augmented_image = replace_background_with_dtd(
                    foreground_rgba,
                    self.dtd_texture_loader,
                    self.dtd_texture_classes,
                    blur_background=True
                )
            else:
                # Random texture from any available class
                all_classes = list(self.dtd_texture_loader.texture_index.keys())
                random_classes = random.sample(all_classes, min(4, len(all_classes)))
                augmented_image = replace_background_with_dtd(
                    foreground_rgba,
                    self.dtd_texture_loader,
                    random_classes,
                    blur_background=True
                )
            
            # Explicitly free memory
            del foreground_rgba, image_array
            
            return augmented_image
        
        except Exception as e:
            print(f"[DreamBooth] Background augmentation failed: {e}. Using original image.")
            return image
    
    def __getitem__(self, index):
        example = {}
        
        # Load instance image
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)
        
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        # Apply DTD background augmentation
        instance_image = self._apply_dtd_background_augmentation(instance_image)
        
        example["instance_images"] = self.image_transforms(instance_image)
        
        # Instance prompt embeddings
        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
        
        # Class images (no background augmentation for regularization images)
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)
            
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            
            example["class_images"] = self.image_transforms(class_image)
            
            # Class prompt embeddings
            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask
        
        return example

class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
