from omegaconf import OmegaConf
import torch
from ldm.models.autoencoder import AutoencoderKL
from ldm.util import instantiate_from_config
from src.datasets.dreambooth_dataset import DreamBoothDataset, tokenize_prompt, PromptDataset
from contextlib import contextmanager, nullcontext
from torch import autocast
from transformers import AutoFeatureExtractor
from PIL import Image
import numpy as np
from transformers import CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from torch.optim import AdamW
import itertools
from diffusers.optimization import get_scheduler
from torch.nn import functional as F
from diffusers.training_utils import compute_snr
from tqdm.auto import tqdm
from diffusers import DDPMScheduler
from pathlib import Path
import torch.nn.utils as nn_utils
from ldm.models.diffusion.ddim import DDIMSampler
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from ldm.modules.diffusionmodules.openaimodel import UNetModel
import os
# logger = get_logger(__name__)
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


class Trainer:
    def __init__(self, args):
        self.args = args
        self.__init_model__()
        self.__init_optimizers__()
        self.__process_prior_reservation__()
        self.__init_dataset__()

    def __process_prior_reservation__(self):
        args = self.args
        class_images_dir = Path(self.args.class_data_dir)
        sampler = DDIMSampler(self.model)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < self.args.num_class_images:
            # torch_dtype = torch.float16 if self.args.device == "cuda" else torch.float32
            # if args.prior_generation_precision == "fp32":
            #     torch_dtype = torch.float32
            # elif args.prior_generation_precision == "fp16":
            #     torch_dtype = torch.float16
            # elif args.prior_generation_precision == "bf16":
            #     torch_dtype = torch.bfloat16
            num_new_images = args.num_class_images - cur_class_images
            print(f"Generating {num_new_images} class images for prior preservation.")
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
            precision_scope = autocast if args.precision == "autocast" else nullcontext
            with torch.no_grad():
                i = 0
                with precision_scope(args.device):
                    with self.model.ema_scope():
                        while i < num_new_images:
                            for example in tqdm(sample_dataloader, desc="Generating class images"):
                                prompts = example["prompt"]
                                uc = self.model.get_learned_conditioning([""] * len(prompts))
                                c = self.model.get_learned_conditioning(prompts)
                                shape = [self.args.C, int(self.args.resolution / self.args.f), int(self.args.resolution / self.args.f)]
                                samples, _ = sampler.sample(
                                    S=50,
                                    conditioning=c,
                                    batch_size=len(prompts),
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=7.5,
                                    unconditional_conditioning=uc,
                                    eta=0.5,
                                )               
                                samples = self.model.decode_first_stage(samples) 
                                x_samples_ddim = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                                print(f"Has nsfw content: {has_nsfw_concept}")
                                safe_img_idx = [i for i, x in enumerate(has_nsfw_concept) if not x]
                                images_to_save = x_checked_image[safe_img_idx]
                                print(f"Images_to save: {len(images_to_save)}, type: {type(images_to_save)}")
                                if len(images_to_save) == 0:
                                    print("smth wrong with nsfw")
                                    break
                                print(f"Images without NSFW content: {len(images_to_save)}")
                                for _, image in enumerate(images_to_save):
                                    image = Image.fromarray((image * 255).round().astype("uint8"))
                                    
                                    image.save(os.path.join(class_images_dir, f"{cur_class_images + i:05}.jpg"))
                                    i += 1
                                if i >= num_new_images:
                                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __init_model__(self):
        model_name = "CompVis/stable-diffusion-v1-4"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        config = OmegaConf.load(self.args.config_path)
        ckpt = torch.load(self.args.ckpt, map_location=self.args.device) 
        model = instantiate_from_config(config.model)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.cuda()
        model.eval()
        # self.unet = UNetModel(image_size=32, 
        #                       in_channels=4, 
        #                       out_channesl=4, 
        #                       model_channels=320,
        #                       attention_resolutions=[4, 2, 1],
        #                       num_res_blocks=2,
        #                       channel_mlt=[1,2,4,4],
        #                       num_heads=8,
        #                       use_spatial_transformer=True,
        #                       transformer_depth=1,
        #                       context_dim=768,
        #                       use_checkpoint=True,
        #                       legacy=False)
        self.unet = model.model.diffusion_model
        self.vae = model.first_stage_model
        self.text_encoder = model.cond_stage_model.transformer.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        self.tokenizer = model.cond_stage_model.tokenizer
        self.model = model

        for param in self.vae.parameters():
            param.requires_grad = False
        # if is_xformers_available():
        #     self.model.enable_xformers_memory_efficient_attention()
        self.noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        

    def __init_optimizers__(self):
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size
            )

        self.optimizer = AdamW(itertools.chain(self.unet.parameters(), self.text_encoder.parameters()),
                               lr=self.args.learning_rate,
                               betas=(self.args.adam_beta1, self.args.adam_beta2),
                               weight_decay=self.args.adam_weight_decay,
                               eps=self.args.adam_epsilon)
        
        self.lr_scheduler = get_scheduler("constant",
                                          optimizer=self.optimizer,
                                          num_warmup_steps=0,
                                          num_cycles=self.args.lr_num_cycles,
                                          num_training_steps=self.args.max_train_steps,
                                          power=self.args.lr_power)
        
        weight_dtype = torch.float32
        if self.args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.weight_dtype = weight_dtype  
        if self.args.mixed_precision == "fp16":
            self.vae = self.vae.to(weight_dtype).half()
        elif self.args.mixed_precision == "bf16":
            self.vae = self.vae.to(weight_dtype).bfloat16()


    def __init_dataset__(self):
        if self.args.instance_data_files is None and self.args.instance_data_num is not None:
            self.args.instance_data_files = []
            random_idx = np.random.choice(len(os.listdir(self.args.instance_data_dir)), self.args.instance_data_num, replace=False)
            filenames = list(os.listdir(self.args.instance_data_dir))
            for idx in random_idx:
                self.args.instance_data_files.append(filenames[idx])
        self.dataset = DreamBoothDataset(
            self.args.instance_data_dir,
            instance_prompt=self.args.instance_prompt,
            class_data_root=self.args.class_data_dir,
            class_prompt=self.args.class_prompt,
            class_num=self.args.num_class_images,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            crop_size=self.args.crop_size,
            category=self.args.category,
            dtd_root_dir=self.args.dtd_root_dir,
            background_augmentation_probability=0.8,
            encoder_hidden_states=None,
            tokenizer_max_length=self.args.tokenizer_max_length,
            instance_data_files=self.args.instance_data_files

        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, with_prior_preservation=self.args.with_prior_preservation)
        )


    def train_epoch(self):
        self.unet.train()
        self.text_encoder.train()

        for step, batch in enumerate(self.dataloader):
            pixel_values = batch["pixel_values"].to(self.weight_dtype).to(self.args.device)
            model_input = self.vae.encode(pixel_values).sample() * 0.1825
            noise = torch.randn_like(model_input).to(self.args.device)  
            bzs, channels, _, _ = model_input.shape

            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bzs,), device=model_input.device)
            timesteps = timesteps.long()
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
            encoder_hidden_states = encode_prompt(
                self.text_encoder,
                batch["input_ids"].to(self.args.device),
                batch["attention_mask"].to(self.args.device),
                text_encoder_use_attention_mask=self.args.text_encoder_use_attention_mask
            )
            if self.unet.in_channels == channels * 2:
                noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

            model_pred = self.unet(noisy_model_input, timesteps, context=encoder_hidden_states)
            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise

            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            if self.args.with_prior_preservation:
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            if self.args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                snr = compute_snr(self.noise_scheduler, timesteps)

                if self.noise_scheduler.config.prediction_type == "v_prediction":
                    divisor = snr + 1
                else:
                    divisor = snr
                mse_loss_weights = (
                    torch.stack([snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / divisor

                )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
            if self.args.with_prior_preservation:
                loss = loss + self.args.prior_loss_weight * prior_loss
            
            loss.backward()
            params_to_clip = itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
            nn_utils.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.args.set_grad_to_none)
            logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
            self.progress_bar.set_postfix(**logs)
            self.progress_bar.update(1)
            self.global_step += 1

            if self.global_step >= self.args.max_train_steps:
                break

            

    def train(self):
        self.global_step = 0
        self.progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=0,
            desc="Steps"
        )
        total_batch_size = self.args.train_batch_size

        print("***** Running training **self.***")
        print(f"  Num examples = {len(self.dataset)}")
        print(f"  Num batches each epoch = {len(self.dataloader)}")
        print(f"  Num Epochs = {self.args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Total optimization steps = {self.args.max_train_steps}")
        for _ in range(0, self.args.num_train_epochs):
            self.train_epoch()
        if not os.path.exists(os.path.dirname(self.args.output_weights_path)):
            os.makedirs(os.path.dirname(self.args.output_weights_path))
        torch.save({"state_dict": self.model.state_dict()}, self.args.output_weights_path)
        if not os.path.exists(os.path.dirname(self.args.output_text_encoder_path)):
            os.makedirs(os.path.dirname(self.args.output_text_encoder_path))
        torch.save(self.text_encoder.state_dict(), self.args.output_text_encoder_path)

