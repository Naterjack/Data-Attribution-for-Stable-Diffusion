#   This code is based on the sample code provided by traker at 
#    https://trak.readthedocs.io/en/latest/quickstart.html
#   and the hugging face diffusers training code since their API does not let you get the loss directly
#    https://github.com/huggingface/diffusers/blob/v0.30.3/examples/text_to_image/train_text_to_image.py       
#

import torch
from trak import TRAKer
from SD1ModelOutput import SD1ModelOutput
from torch.utils.data import DataLoader
from tqdm import tqdm

####CONFIG####
CONVERT_SAFETENSORS_TO_CKPT = False
UPDATE_FEATURIZATION = True
PROFILE_GPU = False
EARLY_EXIT = 0
TRAK_SAVE_DIR = "trak_results_lora"
IS_LORA = True


#Python actually high key stinks
# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils.config import Project_Config, Model_Config, CIFAR_10_Config, LoRA_Model_Config

project_config = Project_Config(
    IS_CUDA = True,
    IS_WINDOWS = False,
)

if IS_LORA:
    model_config = LoRA_Model_Config(
        PROJECT_CONFIG=project_config,
        MODEL_DIR="sd1-cifar10-v2-lora",
        NUM_CHECKPOINTS=10,
        ITERATIONS_PER_CHECKPOINT=10000,
    )
else:
    model_config = Model_Config(
        PROJECT_CONFIG=project_config,
        MODEL_DIR="sd1-cifar10-v2",
        NUM_CHECKPOINTS=10,
        ITERATIONS_PER_CHECKPOINT=10000,
    )

dataset_config = CIFAR_10_Config(new_image_column_name="image",
                                 new_caption_column_name="label_txt")

p = model_config.getModelDirectory()

if IS_LORA:
    tokenizer, text_encoder, vae, _ = model_config.loadModelComponents("stable-diffusion-v1-5/stable-diffusion-v1-5")
    unet = model_config.loadLoRAUnet(p,0)
    ckpts = range(model_config.NUM_CHECKPOINTS)
else:
    ckpts = model_config.loadCheckpoints(p, CONVERT_SAFETENSORS_TO_CKPT)
    tokenizer, text_encoder, vae, unet = model_config.loadModelComponents(p)


#TODO: allow this to be disabled
#import xformers
#unet.enable_xformers_memory_efficient_attention()
#torch.backends.cuda.matmul.allow_tf32 = True

if IS_LORA:
    lora_layers = []
    lora_layers_filter = filter(lambda p: p[1].requires_grad, unet.named_parameters())
    for layer in lora_layers_filter:
        lora_layers.append(layer[0])
else:
    lora_layers = None


#https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(count_parameters(unet))

print(":P")

if project_config.IS_CUDA:
    text_encoder.to("cuda")
    vae.to("cuda")
    unet.to("cuda")

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

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
                grad_wrt=lora_layers
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
        if IS_LORA:
            #TODO: fix the horrid local/global unet duplication going on here
            unet = model_config.loadLoRAUnet(p,ckpt)
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