#   This code is based on the sample code provided by traker at 
#    https://trak.readthedocs.io/en/latest/quickstart.html
#   and the hugging face diffusers training code since their API does not let you get the loss directly
#    https://github.com/huggingface/diffusers/blob/v0.30.3/examples/text_to_image/train_text_to_image.py       
#

import torch
from trak import TRAKer, projectors
from SD1ModelOutput import SD1ModelOutput
from torch.utils.data import DataLoader
from tqdm import tqdm
from TRAK.TRAK_utils import TRAK_Config
from utils.custom_enums import TRAK_Type_Enum, TRAK_Num_Timesteps_Enum, Dataset_Type_Enum, Model_Type_Enum

####CONFIG####
UPDATE_FEATURIZATION = True
DATASET_TYPE = Dataset_Type_Enum.CIFAR10
MODEL_TYPE = Model_Type_Enum.LORA
TRAK_TYPE = TRAK_Type_Enum.DTRAK
TRAK_TIMESTEPS = TRAK_Num_Timesteps_Enum.TEN

###GPU PROFILING###
PROFILE_GPU = False
EARLY_EXIT = 0


#Python actually high key stinks
# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils.config import Project_Config, CIFAR_10_Config, CIFAR_10_Local_Config

project_config = Project_Config(
    IS_CUDA = True,
    IS_WINDOWS = False,
)

trak_config = TRAK_Config(
    project_config=project_config,
    model_type=MODEL_TYPE,
    TRAK_type=TRAK_TYPE,
    dataset_type=DATASET_TYPE,
    TRAK_num_timesteps=TRAK_TIMESTEPS,
)

if DATASET_TYPE == Dataset_Type_Enum.CIFAR10:
    dataset_config = CIFAR_10_Config()
if DATASET_TYPE == Dataset_Type_Enum.CIFAR2:
    dataset_config = CIFAR_10_Local_Config(
        project_config=project_config,
        dataset_type=DATASET_TYPE,
    )

#TODO: allow this to be disabled
#import xformers
#unet.enable_xformers_memory_efficient_attention()
#torch.backends.cuda.matmul.allow_tf32 = True

traker, tokenizer, text_encoder, vae, unet, ckpts = trak_config.load_TRAKer(len(dataset_config.dataset))

train_dataset = dataset_config.preprocess(tokenizer)

# DataLoaders creation:
train_dataloader = DataLoader(
    train_dataset,
    shuffle=False,
    collate_fn=dataset_config.collate_fn,
    batch_size=1,
)

print(traker.num_params_for_grad)

#Scoop whatever VRAM we can because this is going to be a tight fit
import gc
gc.collect()
torch.cuda.empty_cache()

#for model_id, ckpt in enumerate(ckpts): #tqdm

# TRAKer loads the provided checkpoint and also associates

# the provided (unique) model_id with the checkpoint.
weight_dtype = torch.float32

def updateTRAKFeatures():
    i=0
    for model_id, ckpt in enumerate(tqdm(ckpts)):
        if trak_config.model_type == Model_Type_Enum.LORA:
            p = trak_config.model_config.getModelDirectory()
            #TODO: fix the horrid local/global unet duplication going on here
            unet = trak_config.model_config.loadLoRAUnet(p,ckpt)
            traker.load_checkpoint(unet.state_dict(),model_id=model_id)
        else:
            traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in tqdm(train_dataloader):
            if project_config.IS_CUDA:
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
            if EARLY_EXIT:
                i += 1
                if i >= EARLY_EXIT:
                    return
            
        # Tells TRAKer that we've given it all the information, at which point

        # TRAKer does some post-processing to get ready for the next step

        # (scoring target examples).

    traker.finalize_features()

if UPDATE_FEATURIZATION:
    if PROFILE_GPU:
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("trak_featurisation"):
                updateTRAKFeatures()
        
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=30))
        # Print aggregated stats
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

        #prof.export_chrome_trace("trak_trace.json")

        #If profiling outside of Pytorch use the following
        #torch.cuda.cudart().cudaProfilerStart()
        #updateTRAKFeatures()
        #torch.cuda.cudart().cudaProfilerStop()
    else:
        updateTRAKFeatures()