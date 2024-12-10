import os
import torch
from typing import Union
import random
import torch.utils.checkpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms
import numpy as np
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    UNet2DConditionModel,
)


def get_noise_pred_single(unet, latents, t, embeddings):
    noise_pred_single = unet(latents, t, encoder_hidden_states=embeddings)["sample"]
    return noise_pred_single


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, scheduler,
              sample: Union[torch.FloatTensor, np.ndarray]):
    timestep, next_timestep = min(
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample  # ≈ add noise


@torch.no_grad()
def ddim_forward_loop(latent, scheduler, uncond_embeddings, NUM_DDIM_STEPS, unet):
    all_latent = [latent]
    all_times = [0]
    latent = latent.clone().detach()
    for i in range(NUM_DDIM_STEPS):
        t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(unet, latent, t, uncond_embeddings)
        latent = next_step(noise_pred, t, scheduler=scheduler, sample=latent)
        all_latent.append(latent)
        all_times.append(t)
    return all_latent, all_times


def set_xformers(unet):
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        print("xformers is not available.")


def process_img(img_path=None, size=512, resample=Image.BICUBIC, is_training=False):
    if not img_path:
        raise ValueError("img_path is required")
    image = Image.open(img_path)
    image = image.convert("RGB") if image.mode != "RGB" else image
    image_processed = image.resize((size, size), resample=resample)
    if is_training:
        image_processed = transforms.RandomHorizontalFlip(p=0.5)(image_processed)
    image_processed = np.array(image_processed).astype(np.uint8)
    image_processed = (image_processed / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
    img = torch.from_numpy(image_processed).permute(2, 0, 1)  # HWC->CHW
    img = img.unsqueeze(0)
    return img


def prepare_SD_modules(pretrained_model='/media/inspur/disk/yychang_workspace/stable_diffusion_v1_5/stable-diffusion-v1-5/snapshots/module'):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    noise_scheduler = PNDMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    return tokenizer, text_encoder, vae, unet, noise_scheduler

