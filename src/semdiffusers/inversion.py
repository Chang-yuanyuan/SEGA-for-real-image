import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, Tuple, List, Dict
import torch.optim as optim
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from transformers import CLIPTokenizer, CLIPTextModel


@torch.no_grad()
def inverse(test_img_path, device, NUM_DDIM_STEPS, SD_path):
    device = device
    tokenizer, text_encoder, vae, unet, noise_scheduler = prepare_SD_modules(pretrained_model=SD_path)

    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)

    noise_scheduler.set_timesteps(NUM_DDIM_STEPS, device=device)
    img = process_img(img_path=test_img_path,
                      size=512,
                      )
    img = img.to(device)

    z0 = vae.encode(img).latent_dist.mean.detach()
    z0 = z0 * 0.18215

    uncond_text_ids = tokenizer(
        "",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids.to(z0.device)
    uncond_embedding = text_encoder(uncond_text_ids)[0].repeat(z0.shape[0], 1, 1)

    all_latent, all_times = ddim_forward_loop(z0, noise_scheduler, uncond_embedding, NUM_DDIM_STEPS, unet)
    noisy_latents = all_latent[-1]
    return noisy_latents


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    noise_latent = inverse(test_img_path='',
                           device=device,
                           num_DDIM_STEPS=50)
