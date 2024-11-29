import trak
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DDPMScheduler

from typing import Iterable

import trak.modelout_functions

import os
from pathlib import Path


# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from utils.config import Project_Config
from utils.custom_enums import TRAK_Type_Enum, validate_enum

class SD1ModelOutput(trak.modelout_functions.AbstractModelOutput):
    
    noise_scheduler = None
    loss_fn = None

    @staticmethod
    def TRAK_loss(target: torch.Tensor,
                   generated: torch.Tensor):
        return F.mse_loss(target, generated)
    
    @staticmethod
    def DTRAK_loss(target: torch.Tensor,
                   generated: torch.Tensor):
        return F.mse_loss(torch.zeros_like(target), generated)

    def __init__(self,
                 project_config: Project_Config,
                 TRAK_type: TRAK_Type_Enum = TRAK_Type_Enum.TRAK,
                 BASE_MODEL_DIR: str | None = None) -> None:
        #Note that this is the BASE model, not the LoRA fine tuned version, since the LoRA
        #   version doesn't include the noise scheduler
        #super.__init__(self)
        self.TRAK_type = validate_enum(TRAK_type, TRAK_Type_Enum)
        self.project_config = project_config
        f = project_config.folder_symbol
        if BASE_MODEL_DIR is not None:
            p = BASE_MODEL_DIR
        else:
            p = (
                project_config.PWD + f + 
                "models" + f +
                "cifar10" + f +
                "sd1-full" + f
            )
        assert(os.path.isdir(p))
        SD1ModelOutput.noise_scheduler = DDPMScheduler.from_pretrained(p, subfolder="scheduler")

        if TRAK_type == TRAK_Type_Enum.DTRAK:
            SD1ModelOutput.loss_fn = SD1ModelOutput.DTRAK_loss
        else:
            SD1ModelOutput.loss_fn = SD1ModelOutput.TRAK_loss

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

        loss = SD1ModelOutput.loss_fn(target[0], model_pred[0])
        #
        #TRAK
        #
        #loss = F.mse_loss(target[0], model_pred[0])

        #
        #DTRAK
        #
        #loss = F.mse_loss(torch.zeros_like(target[0]), model_pred[0])
        return loss

    def get_out_to_loss_grad(self, model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        return torch.eye(1)