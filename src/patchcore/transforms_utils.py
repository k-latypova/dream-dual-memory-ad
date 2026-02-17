import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image



class ResizeLongestSide:
    """
    Resize image so that the longest side equals target_size,
    preserving aspect ratio.

    Examples:
        >>> resize = ResizeLongestSide(target_size=518)
        >>> img = Image.open('image.jpg')  # 640×480
        >>> resized = resize(img)  # 518×388
    """

    def __init__(self, target_size=518, interpolation=Image.BILINEAR):
        """
        Args:
            target_size (int): Target size for the longest side
            interpolation: PIL interpolation mode (BILINEAR, BICUBIC, etc.)
        """
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image

        Returns:
            PIL Image: Resized image with longest side = target_size
        """
        w, h = img.size

        # Calculate new dimensions
        if h > w:
            # Portrait or square (height >= width)
            new_h = self.target_size
            new_w = int(w * self.target_size / h)
        else:
            # Landscape (width > height)
            new_w = self.target_size
            new_h = int(h * self.target_size / w)

        # Resize with aspect ratio preserved
        resized = TF.resize(img, (new_h, new_w), interpolation=self.interpolation)

        return resized

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(target_size={self.target_size}, '
                f'interpolation={self.interpolation})')


class PadToSquare:
    """
    Pad image to square dimensions (target_size × target_size).

    If image is already larger than target_size, it will NOT be resized,
    only padded to the nearest square that fits.

    Examples:
        >>> pad = PadToSquare(target_size=518)
        >>> img = Image.open('image.jpg')  # 518×388
        >>> padded = pad(img)  # 518×518
    """

    def __init__(self, target_size=518, fill=0, padding_mode='constant'):
        """
        Args:
            target_size (int): Target square size
            fill (int or tuple): Padding fill value
                - int: same value for all channels (0=black, 255=white)
                - tuple: (R, G, B) values for RGB images
            padding_mode (str): Padding mode
                - 'constant': Pad with fill value
                - 'edge': Replicate edge pixels
                - 'reflect': Reflect pixels at boundary
                - 'symmetric': Reflect with boundary pixel included
        """
        self.target_size = target_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image

        Returns:
            PIL Image: Padded square image (target_size × target_size)
        """
        w, h = img.size

        # Calculate padding needed
        pad_w = max(0, self.target_size - w)
        pad_h = max(0, self.target_size - h)

        # Calculate symmetric padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply padding
        if pad_w > 0 or pad_h > 0:
            padded = TF.pad(
                img,
                padding=(pad_left, pad_top, pad_right, pad_bottom),
                fill=self.fill,
                padding_mode=self.padding_mode
            )
        else:
            # No padding needed
            padded = img

        return padded

    def get_padding_info(self, original_size):
        """
        Get padding information for later removal

        Args:
            original_size (tuple): (width, height) of original image

        Returns:
            dict: Padding information
        """
        w, h = original_size
        pad_w = max(0, self.target_size - w)
        pad_h = max(0, self.target_size - h)

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        return {
            'padding': (pad_top, pad_bottom, pad_left, pad_right),
            'original_size': (h, w),  # (height, width) for tensor indexing
            'padded_size': (self.target_size, self.target_size)
        }

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(target_size={self.target_size}, '
                f'fill={self.fill}, '
                f'padding_mode={self.padding_mode})')


class PadToSquareTensor:
    """
    Pad tensor to square dimensions in normalized feature space.

    This version pads AFTER ToTensor and Normalize transforms,
    using zero values in normalized space (better than normalized black pixels).

    Examples:
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        ...                         std=[0.229, 0.224, 0.225]),
        ...     PadToSquareTensor(target_size=518)
        ... ])
    """

    def __init__(self, target_size=518, fill=0.0):
        """
        Args:
            target_size (int): Target square size
            fill (float): Padding fill value in normalized space
                - 0.0: True zero (recommended)
                - Other values: Custom padding values
        """
        self.target_size = target_size
        self.fill = fill

    def __call__(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor (C, H, W)

        Returns:
            torch.Tensor: Padded tensor (C, target_size, target_size)
        """
        c, h, w = tensor.shape

        # Calculate padding needed
        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)

        # Calculate symmetric padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding in feature space
        if pad_h > 0 or pad_w > 0:
            padded = torch.nn.functional.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=self.fill
            )
        else:
            padded = tensor

        return padded

    def get_padding_info(self, original_size):
        """
        Get padding information for later removal

        Args:
            original_size (tuple): (C, H, W) of tensor before padding

        Returns:
            dict: Padding information
        """
        _, h, w = original_size
        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return {
            'padding': (pad_top, pad_bottom, pad_left, pad_right),
            'original_size': (h, w),
            'padded_size': (self.target_size, self.target_size)
        }

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(target_size={self.target_size}, '
                f'fill={self.fill})')


# =============================================================================
# Pre-configured transform pipelines
# =============================================================================

def create_resize_pad_transform(target_size=518, 
                                normalize=True,
                                padding_mode='constant',
                                fill=0):
    """
    Create a complete preprocessing pipeline with separate resize and pad steps.

    Args:
        target_size (int): Target square size
        normalize (bool): Whether to apply ImageNet normalization
        padding_mode (str): Padding mode ('constant', 'edge', 'reflect', 'symmetric')
        fill (int): Fill value for padding (0=black, 255=white)

    Returns:
        transforms.Compose: Complete preprocessing pipeline

    Examples:
        >>> transform = create_resize_pad_transform(target_size=518)
        >>> img = Image.open('image.jpg')
        >>> tensor = transform(img)  # (3, 518, 518)
    """
    transform_list = [
        ResizeLongestSide(target_size=target_size),
        PadToSquare(target_size=target_size, fill=fill, padding_mode=padding_mode),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(transform_list)


def create_resize_pad_feature_space(target_size=518, 
                                    normalize=True,
                                    fill=0.0):
    """
    Create preprocessing pipeline with padding in feature space (after normalization).

    This is the RECOMMENDED approach as it pads with true zeros in normalized
    space rather than normalized black pixels.

    Args:
        target_size (int): Target square size
        normalize (bool): Whether to apply normalization
        fill (float): Fill value in normalized space (default: 0.0)

    Returns:
        transforms.Compose: Complete preprocessing pipeline

    Examples:
        >>> transform = create_resize_pad_feature_space(target_size=518)
        >>> img = Image.open('image.jpg')
        >>> tensor = transform(img)  # (3, 518, 518) with zero padding
    """
    transform_list = [
        ResizeLongestSide(target_size=target_size),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        )

    transform_list.append(
        PadToSquareTensor(target_size=target_size, fill=fill)
    )

    return transforms.Compose(transform_list)


# =============================================================================
# Helper functions
# =============================================================================

def get_transform_metadata(original_size, target_size=518):
    """
    Calculate transformation metadata without actually transforming

    Useful for tracking padding info for later removal

    Args:
        img (PIL Image): Input image
        target_size (int): Target square size

    Returns:
        dict: Metadata including original size, resize dimensions, and padding

    Examples:
        >>> img = Image.open('image.jpg')
        >>> metadata = get_transform_metadata(img, target_size=518)
        >>> print(metadata['padding'])  # (top, bottom, left, right)
    """
    w, h = original_size

    # Calculate resize dimensions
    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)

    # Calculate padding
    pad_w = max(0, target_size - new_w)
    pad_h = max(0, target_size - new_h)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    return {
        'original_size': (w, h),
        'resized_size': (new_w, new_h),
        'final_size': (target_size, target_size),
        'padding': (pad_top, pad_bottom, pad_left, pad_right),
        'aspect_ratio': w / h
    }


def remove_padding(tensor, padding_info):
    """
    Remove padding from tensor using padding info

    Args:
        tensor (torch.Tensor): Padded tensor (C, H, W) or (H, W)
        padding_info (dict): Dict with 'padding' key containing (top, bottom, left, right)

    Returns:
        torch.Tensor: Unpadded tensor

    Examples:
        >>> padded_tensor = transform(img)  # (3, 518, 518)
        >>> metadata = get_transform_metadata(img, target_size=518)
        >>> unpadded = remove_padding(padded_tensor, metadata)
    """
    pad_top, pad_bottom, pad_left, pad_right = padding_info['padding']

    if tensor.dim() == 3:
        # (C, H, W)
        c, h, w = tensor.shape
        crop_h = slice(pad_top, h - pad_bottom if pad_bottom > 0 else None)
        crop_w = slice(pad_left, w - pad_right if pad_right > 0 else None)
        return tensor[:, crop_h, crop_w]

    elif tensor.dim() == 2:
        # (H, W)
        h, w = tensor.shape
        crop_h = slice(pad_top, h - pad_bottom if pad_bottom > 0 else None)
        crop_w = slice(pad_left, w - pad_right if pad_right > 0 else None)
        return tensor[crop_h, crop_w]

    else:
        raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")