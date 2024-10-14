import trak
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DDPMScheduler

from typing import Iterable

import trak.modelout_functions

MODEL_DIR = "sd1-cifar10-v2"
IS_WINDOWS = False
import os
from pathlib import Path


class SD1ModelOutput(trak.modelout_functions.AbstractModelOutput):

    if IS_WINDOWS: 
        folder_symbol = "\\" 
    else: #Assume UNIX-based
        folder_symbol = "/"

    pwd = str(Path(__file__).resolve().parent.parent)
    p = pwd + folder_symbol + MODEL_DIR + folder_symbol
    assert(os.path.isdir(p))
    
    noise_scheduler = DDPMScheduler.from_pretrained(p, subfolder="scheduler")


    def __init__(self) -> None:
        super.__init__()

    @staticmethod
    def get_output(model: torch.nn.Module,
            weights: Iterable[Tensor],
            buffers: Iterable[Tensor],
            latents: Tensor,
            encoder_hidden_states: Tensor,) -> Tensor:
        
        latents = latents.unsqueeze(0)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
        #This code is (mostly) taken from the original huggingface training code, since their API
        # does not let you get the loss directly
        # The original code can be found here: https://github.com/huggingface/diffusers/blob/v0.30.3/examples/text_to_image/train_text_to_image.py
        noise_scheduler = SD1ModelOutput.noise_scheduler
        unet = model

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = torch.func.functional_call(unet, (weights, buffers), (noisy_latents, timesteps, encoder_hidden_states), {"return_dict":False})
        target = target.unsqueeze(0)
        loss = F.mse_loss(target[0], model_pred[0])
        return loss

    def get_out_to_loss_grad(model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        return torch.eye(1)