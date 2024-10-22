#   This code is based on the sample code provided by traker at 
#    https://trak.readthedocs.io/en/latest/quickstart.html
#   and the hugging face diffusers training code since their API does not let you get the loss directly
#    https://github.com/huggingface/diffusers/blob/v0.30.3/examples/text_to_image/train_text_to_image.py       
#

import torch
#from pathlib import Path
from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from torchvision import transforms
import os
from trak import TRAKer
from SD1ModelOutput import SD1ModelOutput
from torch.utils.data import DataLoader
from tqdm import tqdm

####CONFIG####
CONVERT_SAFETENSORS_TO_CKPT = False
CUDA = True
UPDATE_FEATURIZATION = True
IS_WINDOWS = False
NUM_CHECKPOINTS = 10
ITERATIONS_PER_CHECKPOINT = 10000
TRAK_SAVE_DIR = "trak_results_v2"
MODEL_DIR = "sd1-cifar10-v2"

from utils.config import Project_Config, Model_Config, CIFAR_10_Config

project_config = Project_Config(
    CUDA = True,
    IS_WINDOWS = False,
)

model_config = Model_Config(
    PROJECT_CONFIG=project_config,
    MODEL_DIR=MODEL_DIR,
    NUM_CHECKPOINTS=NUM_CHECKPOINTS,
)

dataset_config = CIFAR_10_Config(new_image_column_name="image",
                                 new_caption_column_name="label_txt")

p = model_config.getModelDirectory()
ckpts = model_config.loadCheckpoints(p, CONVERT_SAFETENSORS_TO_CKPT=CONVERT_SAFETENSORS_TO_CKPT)

tokenizer, text_encoder, vae, unet = model_config.loadModelComponents(p)

text_encoder.to("cuda")
vae.to("cuda")
unet.to("cuda")

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
#unet.train()

train_dataset = dataset_config.preprocess(tokenizer)

# DataLoaders creation:
train_dataloader = DataLoader(
    train_dataset,
    shuffle=False,
    collate_fn=dataset_config.collate_fn,
    batch_size=1,
)

traker = TRAKer(model=unet,
                task=SD1ModelOutput,
                train_set_size=len(train_dataloader.dataset),
                save_dir=TRAK_SAVE_DIR,
                proj_max_batch_size=8, #Default 32, requires an A100 apparently
                proj_dim=1024,
                )

#Scoop whatever VRAM we can because this is going to be a tight fit
import gc
gc.collect()
torch.cuda.empty_cache()

#for model_id, ckpt in enumerate(ckpts): #tqdm

# TRAKer loads the provided checkpoint and also associates

# the provided (unique) model_id with the checkpoint.
#model_id = 0
#traker.load_checkpoint(ckpt, model_id=model_id)
weight_dtype = torch.float32

if UPDATE_FEATURIZATION:
    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in tqdm(train_dataloader):
            if CUDA:
                batch = [x.cuda() for y,x in batch.items()]
            else:
                batch = [x for y,x in batch.items()]
            image = batch[0]
            tokens = batch[1]
            # Convert images to latent space
            latents = vae.encode(image.to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            encoder_hidden_states = text_encoder(tokens, return_dict=False)[0]

            batch = [latents, encoder_hidden_states]

            # TRAKer computes features corresponding to the batch of examples,
            # using the checkpoint loaded above.

            traker.featurize(batch=batch, num_samples=batch[0].shape[0])


        # Tells TRAKer that we've given it all the information, at which point

        # TRAKer does some post-processing to get ready for the next step

        # (scoring target examples).

    traker.finalize_features()