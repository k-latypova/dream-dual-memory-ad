import argparse
import gc
import os
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
import torch
import numpy as np
from omegaconf import OmegaConf
import cv2 
from PIL import Image
from tqdm.auto import tqdm, trange
from einops import rearrange
import random
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from contextlib import nullcontext
from src.utils.datasets_utils import get_class_names
import torch.nn.functional as F
import json
import pandas as pd
from src.utils.image_processing import reinhard_color_transfer, adjust_exposure, put_watermark, crop_mask_with_padding, resize_pil_img_with_cv2
from skimage.exposure import match_histograms
from src.utils.mask_utils import object_blob_mask


def postprocess_rgb_img(original_image_np, image_np, mask_np, cropbox, cropped_img: np.array):
    image_np = (image_np * 255).astype(np.uint8)
    image_np = np.clip(image_np, 0, 255)
    image_np = rearrange(image_np, 'c h w -> h w c')
    color_image = reinhard_color_transfer(cropped_img, image_np)
    color_image = adjust_exposure(color_image,1.05)
    #color_image = image_np
    
    x1, y1, x2, y2 = cropbox
    # sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
    # path = "LapSRN_x8.pb"
    
    # sr.readModel(path)
    
    # sr.setModel("lapsrn",8)
    
    # color_image = sr.upsample(color_image)
 
 
    # Resized image
    color_image = cv2.resize(color_image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    # sigma = 1.5          # increase to sharpen broader edges
    # amount = 0.8         # increase to strengthen sharpening
    # blur = cv2.GaussianBlur(color_image, (0,0), sigmaX=sigma)
    # color_image = cv2.addWeighted(color_image, 1.0 + amount, blur, -amount, 0)

    original_image_norm = original_image_np.astype(np.float32) / 255.0
    original_image_norm = rearrange(original_image_norm, 'h w c -> c h w')

    image_to_paste = Image.fromarray(color_image)
    original_image = Image.fromarray(original_image_np, 'RGB')
    original_image.paste(image_to_paste, box=cropbox)
    


    #blur mask
    
    blurred_mask = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=5.0, sigmaY=5.0)
    blurred_mask_img = Image.fromarray(blurred_mask, mode='L')
    blurred_mask_np = np.array(blurred_mask_img).astype(np.float32) / 255.0
    blurred_mask_np = rearrange(blurred_mask_np, 'h w -> 1 h w')

    

    
    
    #return original_image
    pasted_crop_np = np.array(original_image).astype(np.float32) / 255.0
    pasted_crop_np = rearrange(pasted_crop_np, 'h w c -> c h w')
    final_img_np = blurred_mask_np * pasted_crop_np + (1 - blurred_mask_np) * original_image_norm
    final_img_np = np.clip(final_img_np, 0, 1)
    final_img_np = rearrange(final_img_np, 'c h w -> h w c') * 255.0
    final_img = Image.fromarray(final_img_np.astype(np.uint8))
    return original_image, final_img


def postprocess_rgb_img__(original_image_np, image_np, mask_np, cropbox, cropped_img: np.array):
    image_np = (image_np * 255).astype(np.uint8)
    image_np = np.clip(image_np, 0, 255)
    image_np = rearrange(image_np, 'c h w -> h w c')
    color_image = reinhard_color_transfer(cropped_img, image_np)
    color_image = image_np
    
    x1, y1, x2, y2 = cropbox
    color_image = cv2.resize(color_image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    sigma = 1.5          # increase to sharpen broader edges
    amount = 0.8         # increase to strengthen sharpening
    blur = cv2.GaussianBlur(color_image, (0,0), sigmaX=sigma)
    color_image = cv2.addWeighted(color_image, 1.0 + amount, blur, -amount, 0)

    original_image_norm = original_image_np.astype(np.float32) / 255.0
    original_image_norm = rearrange(original_image_norm, 'h w c -> c h w')


    #blur mask
    
    # blurred_mask = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=7, sigmaY=7, 
    #                                 borderType=cv2.BORDER_DEFAULT)
    #blurred_mask = skimage.exposure.rescale_intensity(blurred_mask, in_range=(127.5,255), out_range=(0,255))
    blurred_mask_img = Image.fromarray(mask_np, mode='L')
    blurred_mask_np = np.array(blurred_mask_img).astype(np.float32) / 255.0
    blurred_mask_np = rearrange(blurred_mask_np, 'h w -> 1 h w')

    image_to_paste = Image.fromarray(color_image)
    original_image = Image.fromarray(original_image_np, 'RGB')
    original_image.paste(image_to_paste, box=cropbox)

    #return original_image
    pasted_crop_np = np.array(original_image).astype(np.float32) / 255.0
    pasted_crop_np = rearrange(pasted_crop_np, 'h w c -> c h w')
    final_img_np = blurred_mask_np * pasted_crop_np + (1 - blurred_mask_np) * original_image_norm
    final_img_np = np.clip(final_img_np, 0, 1)
    final_img_np = rearrange(final_img_np, 'c h w -> h w c') * 255.0
    final_img = Image.fromarray(final_img_np.astype(np.uint8))
    return final_img

def postprocess_rgb_img_2(original_image_np, image_np, mask_np, cropbox, cropped_img: np.array):
    image_np = (image_np * 255).astype(np.uint8)
    image_np = np.clip(image_np, 0, 255)
    image_np = rearrange(image_np, 'c h w -> h w c')
    color_image = reinhard_color_transfer(cropped_img, image_np)
    color_image = adjust_exposure(color_image,1.05)
    #color_image = image_np
    
    x1, y1, x2, y2 = cropbox
    # sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
    # path = "LapSRN_x8.pb"
    
    # sr.readModel(path)
    
    # sr.setModel("lapsrn",8)
    
    # color_image = sr.upsample(color_image)
 
 
    # Resized image
    color_image = cv2.resize(color_image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
    # sigma = 1.5          # increase to sharpen broader edges
    # amount = 0.8         # increase to strengthen sharpening
    # blur = cv2.GaussianBlur(color_image, (0,0), sigmaX=sigma)
    # color_image = cv2.addWeighted(color_image, 1.0 + amount, blur, -amount, 0)

    original_image_norm = original_image_np.astype(np.float32) / 255.0
    original_image_norm = rearrange(original_image_norm, 'h w c -> c h w')

    image_to_paste = Image.fromarray(color_image)
    original_image = Image.fromarray(original_image_np, 'RGB')
    original_image.paste(image_to_paste, box=cropbox)
    


    #blur mask
    
    blurred_mask = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=5.0, sigmaY=5.0)
    blurred_mask_img = Image.fromarray(blurred_mask, mode='L')
    blurred_mask_np = np.array(blurred_mask_img).astype(np.float32) / 255.0
    blurred_mask_np = rearrange(blurred_mask_np, 'h w -> 1 h w')

    

    
    
    #return original_image
    pasted_crop_np = np.array(original_image).astype(np.float32) / 255.0
    pasted_crop_np = rearrange(pasted_crop_np, 'h w c -> c h w')
    final_img_np = blurred_mask_np * pasted_crop_np + (1 - blurred_mask_np) * original_image_norm
    final_img_np = np.clip(final_img_np, 0, 1)
    final_img_np = rearrange(final_img_np, 'c h w -> h w c') * 255.0
    final_img = Image.fromarray(final_img_np.astype(np.uint8))
    return original_image, final_img


def postprocess_greyscale_img(original_image_np, image_np, mask_np, cropbox, cropped_img: np.array):
    image_np = (image_np * 255).astype(np.uint8)
    image_np = np.clip(image_np, 0, 255)
    image_np = rearrange(image_np, 'c h w -> h w c')
    original_image = Image.fromarray(original_image_np, 'RGB').convert("L")
    original_image_np = np.array(original_image)

    cropped_greyscale = Image.fromarray(cropped_img).convert("L")
    image_greyscale = Image.fromarray(image_np).convert("L")
    matched_img = match_histograms(np.array(image_greyscale), np.array(cropped_greyscale))

    x1, y1, x2, y2 = cropbox
    matched_img = cv2.resize(matched_img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)

    original_image_norm = original_image_np.astype(np.float32) / 255.0


    #blur mask
    #blurred_mask = cv2.GaussianBlur(mask_np, (21, 21), 13)
    blurred_mask_img = Image.fromarray(mask_np)
    blurred_mask_np = np.array(blurred_mask_img).astype(np.float32) / 255.0
    #blurred_mask_np = rearrange(blurred_mask_np, 'h w -> 1 h w')

    image_to_paste = Image.fromarray(matched_img)
    original_image.paste(image_to_paste, box=cropbox)
    pasted_crop_np = np.array(original_image).astype(np.float32) / 255.0
    final_img_np = blurred_mask_np * pasted_crop_np + (1 - blurred_mask_np) * original_image_norm
    final_img_np = np.clip(final_img_np, 0, 1) * 255.0
    final_img = Image.fromarray(final_img_np.astype(np.uint8), "L")
    return final_img


def generate_from_img_mask(args, model, img, mask, prompts, sampler):
    cropped_img, cropped_mask, crop_box = crop_mask_with_padding(img, mask, ratio=args.ratio, target_img_size=(args.W, args.W))
    cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
    cropped_mask_np[cropped_mask_np < 0.5] = 0.0
    cropped_mask_np[cropped_mask_np >= 0.5] = 1.0
    cropped_mask_tensor = torch.from_numpy(cropped_mask_np).float().to(args.device)
    print(f"Image shape: {np.array(cropped_img).shape}, Mask shape: {cropped_mask_np.shape}")
    cropped_img_arr = np.array(cropped_img).transpose(2, 0, 1) / 255.0
    cropped_img_tensor = torch.from_numpy(cropped_img_arr).float().unsqueeze(0).to(args.device)
    prior = model.first_stage_model.encode(cropped_img_tensor)
    init_latent = model.get_first_stage_encoding(prior).to(args.device)

    mask_cc = torch.nn.functional.interpolate(
        cropped_mask_tensor.unsqueeze(0).unsqueeze(0),  # Add channel dimension
        size=(int(args.H // args.f), int(args.W // args.f)),
        mode='bilinear',
        align_corners=False
    )
    
    for n in trange(args.n_iter, desc="Sampling iterations"):
        # Clear memory between iterations
        if n % 2 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        uc = None
        if args.gaussian_scale > 0.0:
            noise = torch.randn_like(init_latent, device=args.device) * args.gaussian_scale
        else:
            noise = None
        if args.scale != 1.0:
            uc = model.get_learned_conditioning(args.batch_size * [""], 0, args)
        c = []
        for prompt in prompts:
            c_i = model.get_learned_conditioning([prompt], 0, args)
            c.append(c_i)
        c = torch.stack(c).reshape(args.batch_size, c[0].shape[-2], c[0].shape[-1])
        sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)
        t_enc = int(args.strength * args.ddim_steps) - 1
        if not args.plms:
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*args.batch_size).to(args.device), 
                                              noise=noise)
            samples_ddim = sampler.mask_decode(z_enc, c, t_enc, init_latent, (1-mask_cc),
                                               unconditional_guidance_scale=args.scale,
                                               unconditional_conditioning=uc,)
        else:
            shape = [args.C, args.H // args.f, args.W // args.f]
            samples_ddim, _ = sampler.sample(
                                            S=args.ddim_steps, 
                                            conditioning=c,
                                            batch_size=args.batch_size,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=uc,
                                            eta=0.0,
                                            x0=init_latent,
                                            mask=(1-mask_cc),
                                        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)
            
        
        # Save samples
        for i, x_sample in enumerate(x_samples_ddim):
            if args.greyscale:
                final_img = postprocess_greyscale_img(np.array(img), x_sample.cpu().numpy(), 
                                                      np.array(mask), crop_box, np.array(cropped_img))
            else:
                final_img = postprocess_rgb_img(np.array(img), x_sample.cpu().numpy(), 
                                                np.array(mask), crop_box, np.array(cropped_img))
    return final_img



def generate_anomalies(args):
    seed_everything(args.seed)
    config = OmegaConf.load(args.config_path)
    ckpt = torch.load(args.ckpt, map_location=args.device) 
    model = instantiate_from_config(config.model)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.cuda()
    model.eval()
    del ckpt

    if not args.plms:
        sampler = DDIMSampler(model)
    else:
        sampler = PLMSSampler(model)
        args.ddim_eta = 0.0
    precision_scope = torch.autocast if args.precision == "autocast" else nullcontext

    samples_path = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_path, exist_ok=True)
    masks_path = os.path.join(args.output_dir, "masks")
    os.makedirs(masks_path, exist_ok=True)

    base_count = len(os.listdir(samples_path))
    prompt: str = args.prompt
    if args.anomaly_class in prompt:
        token_idx = prompt.split(" ").index(args.anomaly_class)
    else:
        token_idx = args.token_idx
    
    prompts = [prompt] * args.batch_size

    src_images = []

    images_path = os.path.join(args.data_dir, args.anomaly_class, "train", "good")
    for img in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, img)):
            src_images.append(os.path.join(images_path, img))
       
        
    #masks = torch.stack(masks)
    use_object_mask = not args.mask_anywhere


    with open(os.path.join(args.output_dir, "params.json"), "w") as f:
        params = {
            "prompt": prompt,
            "gaussian_scale": args.gaussian_scale,
            "ddim_steps": args.ddim_steps,
            "num_samples": args.num_samples,
            "loaded_embedding": args.loaded_embeddings,
        }
        json.dump(params, f, indent=4)

    metadata = []

    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    mask_idx = 0

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for img_idx, img_src_path in enumerate(src_images):
                    img = Image.open(img_src_path).convert("RGB")
                    print(f"Processing image {img_idx+1}/{len(src_images)}")
                    for _ in tqdm(range(args.num_masks)):
                        seed = np.random.choice(1000)
                        blob_ratio_idx = np.random.choice(len(args.mask_generation_ratio))
                        blob_ratio = args.mask_generation_ratio[blob_ratio_idx]
                        loaded_embedding_idx = min(blob_ratio_idx, len(args.loaded_embeddings)-1)
                        args.loaded_embedding = args.loaded_embeddings[loaded_embedding_idx]

                        
                        mask = object_blob_mask(img_src_path, seed=seed, out_width=img.width, out_height=img.height, 
                                                blob_ratio=blob_ratio, 
                                                use_object_mask=use_object_mask,
                                                object_mask_func_version=args.object_mask_func_version,
                                                bg_color=args.bg_color)
                        mask_file_path = os.path.join(masks_path, f"mask_{mask_idx:05}.png")
                        mask.save(mask_file_path)
                        mask_idx += 1
                        mask = Image.open(mask_file_path).convert("L")

                        cropped_img, cropped_mask, crop_box = crop_mask_with_padding(img, mask, ratio=args.ratio, target_img_size=(args.W, args.W))
                        cropped_mask_np = np.array(cropped_mask).astype(np.float32) / 255.0
                        cropped_mask_np[cropped_mask_np < 0.5] = 0.0
                        cropped_mask_np[cropped_mask_np >= 0.5] = 1.0
                        cropped_mask_tensor = torch.from_numpy(cropped_mask_np)[None, None, ...]
                        mask_cc = torch.nn.functional.interpolate(cropped_mask_tensor, (int(args.H // args.f), int(args.W // args.f)), mode='nearest').to(args.device)
                        cropped_img_arr = np.array(cropped_img).transpose(2, 0, 1) / 255.0
                        cropped_img_tensor = torch.from_numpy(cropped_img_arr).float().unsqueeze(0).to(args.device)
                        prior = model.first_stage_model.encode(cropped_img_tensor)
                        init_latent = model.get_first_stage_encoding(prior).to(args.device)

                        
                        for n in trange(args.n_iter, desc="Sampling iterations"):
                            # Clear memory between iterations
                            if n % 2 == 0:
                                torch.cuda.empty_cache()
                                gc.collect()

                            uc = None
                            if args.scale != 1.0:
                                uc = model.get_learned_conditioning(args.batch_size * [""], 0, args)
                            c = []
                            for prompt in prompts:
                                c_i = model.get_learned_conditioning([prompt], 0, args, token_idx)
                                c.append(c_i)
                            c = torch.stack(c).reshape(args.batch_size, c[0].shape[-2], c[0].shape[-1])
                            sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)
                            strength = np.random.choice(args.strength)
                            t_enc = int(strength * args.ddim_steps) - 1
                            if not args.plms:
                                if args.strength != 1.0:
                                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*args.batch_size).to(args.device))
                                    samples_ddim = sampler.mask_decode(z_enc, c, t_enc, init_latent, (1-mask_cc),
                                                                    unconditional_guidance_scale=args.scale,
                                                                    unconditional_conditioning=uc,)
                                else:
                                    shape = [args.C, args.H // args.f, args.W // args.f]
                                    samples_ddim, _ = sampler.sample(
                                        S=args.ddim_steps, 
                                        conditioning=c,
                                        batch_size=args.batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=args.scale,
                                        unconditional_conditioning=uc,
                                        eta=args.ddim_eta,                            
                                        x0=init_latent,
                                        mask=1-mask_cc
                                    )
                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                
                            else:
                                shape = [args.C, args.H // args.f, args.W // args.f]
                                samples_ddim, _ = sampler.sample(
                                                                S=args.ddim_steps, 
                                                                conditioning=c,
                                                                batch_size=args.batch_size,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=args.scale,
                                                                unconditional_conditioning=uc,
                                                                eta=0.0,
                                                                x0=init_latent,
                                                                mask=(1-mask_cc),
                                                            )
                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)
                                
                            
                            # Save samples
                            for i, x_sample in enumerate(x_samples_ddim):
                                if args.greyscale:
                                    final_img = postprocess_greyscale_img(np.array(img), x_sample.cpu().numpy(), 
                                                                        np.array(mask), crop_box, np.array(cropped_img))
                                else:
                                    # final_img = rearrange(x_samples_ddim[0].cpu().numpy(), 'c h w -> h w c') * 255.0
                                    # final_img = Image.fromarray(final_img.astype(np.uint8))
                                    final_img = postprocess_rgb_img(np.array(img), x_sample.cpu().numpy(), 
                                                                    np.array(mask), crop_box, np.array(cropped_img))
    
                                # final_img = generate_from_img_mask(args, model, img, mask, prompts, sampler)
                                sample_filename = f"sample_{base_count:05}.png"
                                final_img[1].save(os.path.join(samples_path, sample_filename))
                                metadata.append((img_src_path, mask_file_path, sample_filename))
                                metadata_df = pd.DataFrame(metadata, columns=["normal", "mask_idx", "sample_idx"])
                                metadata_df.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
                                base_count += 1

                    torch.cuda.empty_cache()
                    gc.collect()
    
    metadata_df = pd.DataFrame(metadata, columns=["normal", "mask_idx", "sample_idx"])
    metadata_df.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
    print(f"Saved samples to {samples_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DREAM-AD")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Directory containing the data files",
    )
    parser.add_argument("--anomaly_class", type=str, default=0, help="Anomaly class")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name") 
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="Directory containing the model files",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument("--ddim_steps",
                        type=int,
                        default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate for each input",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument("--plms",
                        action="store_true",
                        help="Use PLMS sampling instead of DDIM")
    parser.add_argument("--gaussian_scale",
                        type=float,
                        default=0.0,
                        help="Gaussian scale for noise addition")    
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter for sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="Number of iterations for sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )

    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="Batch size per prompt")

    parser.add_argument(
        "--loaded_embeddings",
        type=str,
        nargs="+",
        required=True,
        help="Path to the loaded embedding file",
    )
    parser.add_argument(
        "--token_idx", type=int, default=0, help="Which token to replace with generated embedding"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Directory to save the generated images",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to use for image generation. If not provided, will be generated based on the anomaly class.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        nargs="+",
        default=1.0,
        help="Strength of the effect applied to the normal images (1.0 means full effect, 0.0 means no effect)",
    )

    parser.add_argument(
        "--greyscale",
        action="store_true",
        help="Generate greyscale images instead of RGB",
    )
    parser.add_argument("--ratio", type=float, default=0.075, help="Aspect ratio of the generated images")
    parser.add_argument("--mask_generation_ratio", nargs="+", type=float, required=True, default="Ratio of generated mask")
    parser.add_argument("--src-images-metadata", default=None, type=str, help="Path to the JSON file containing metadata for source images and masks")
    parser.add_argument("--num_masks", type=int, default=20, help="Number of masks to generate per src image")
    parser.add_argument("--mask_anywhere", action="store_true")
    parser.add_argument("--object_mask_func_version", type=int, default=2, help="Which function to use to find object on the image")
    parser.add_argument("--bg_color", nargs="+", type=int, default=None, help="Background color to use when generating masks")
    args = parser.parse_args()

    generate_anomalies(args)
    