from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
weight_dtype = torch.float32

#Python actually high key stinks
# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

NUM_TIMESTEPS_TO_SAMPLE = 10

'''

'''
def LDS(attribution_scores_tau: np.array, dataloader_X: DataLoader, dataloader_X_size: int):
    from utils.config import CIFAR_10_Config, Project_Config, Model_Config
    DATASET_NAME = "cifar-10"
    dataset_config = CIFAR_10_Config()
    project_config = Project_Config()

    NUM_SUBSETS = 32

    NUM_CLASSES = 10
    ROWS_PER_CLASS = int(dataset_config.dataset.num_rows/ NUM_CLASSES)

    SUBSET_SIZE_ALPHA = 0.5
    SUBSET_SIZE_TOTAL = int(dataset_config.dataset.num_rows * SUBSET_SIZE_ALPHA)
    ROWS_PER_CLASS_SUBSET = int(ROWS_PER_CLASS*SUBSET_SIZE_ALPHA)

    assert(SUBSET_SIZE_TOTAL == ROWS_PER_CLASS_SUBSET * NUM_CLASSES)
    #print(f"Rows per class per subset {ROWS_PER_CLASS_SUBSET}")

    #indices_cache_file_name="cifar10sortedbyclass")
    dataset_config.dataset = dataset_config.dataset.sort(column_names=dataset_config.caption_column)

    full_dataset_indicies = dataset_config.dataset._indices[0].to_numpy()

    from pathlib import Path

    dataset_base_path = (project_config.PWD + project_config.folder_symbol +
                "datasets" + project_config.folder_symbol +
                DATASET_NAME + project_config.folder_symbol)
    
    subset_matrix = parseREADME(dataset_base_path+"README.md", 
                full_dataset_indicies, 
                NUM_SUBSETS,
                ROWS_PER_CLASS_SUBSET,
                NUM_CLASSES)
    
    assert(subset_matrix[2,0] == full_dataset_indicies[2638])

    DATASET_NAME = "sd1-cifar10-v2" #Shhh dw about it

    models_base_path = (project_config.PWD + project_config.folder_symbol +
                "LDS" + project_config.folder_symbol +
                DATASET_NAME + project_config.folder_symbol)

    MOF_matrix = np.zeros((NUM_SUBSETS,dataloader_X_size))

    
    for i in tqdm(range(1,NUM_SUBSETS)):
        ### Load Model
        model_path = models_base_path + str(i)
        model_config = Model_Config(PROJECT_CONFIG=project_config,
                                MODEL_DIR=model_path)
        tokenizer, text_encoder, vae, unet = model_config.loadModelComponents(model_path)
        if project_config.IS_CUDA:
            text_encoder.to("cuda")
            vae.to("cuda")
            unet.to("cuda")
        
        batch_counter=0
        for batch in dataloader_X:
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
            
            #latents = latents.unsqueeze(0)
            #encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            #This code is (mostly) taken from the original huggingface training code, since their API
            # does not let you get the loss directly
            # The original code can be found here: https://github.com/huggingface/diffusers/blob/v0.30.3/examples/text_to_image/train_text_to_image.py

            noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

            step = noise_scheduler.config.num_train_timesteps // NUM_TIMESTEPS_TO_SAMPLE
            losses = np.zeros((NUM_TIMESTEPS_TO_SAMPLE))
            timestep_count = 0
            for timestep_int in range(0,noise_scheduler.config.num_train_timesteps,step):
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                ## Sample a random timestep for each image
                #timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                #timesteps = timesteps.long()

                timesteps = torch.tensor(timestep_int)
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
                model_pred = unet.forward(noisy_latents, timesteps, encoder_hidden_states)

                target = target.unsqueeze(0)

                #
                #TRAK
                #
                losses[timestep_count] = F.mse_loss(target[0], model_pred[0]).cpu().detach().numpy()
                timestep_count += 1
            #print(f"losses shape: {losses.shape}")
            MOF_matrix[i,batch_counter] = np.average(losses)
            batch_counter += 1
    

    g_scores = np.zeros((NUM_SUBSETS, dataloader_X_size))
    for subset in range(1,NUM_SUBSETS):
        for example in range(dataloader_X_size):
            for training_data_index in subset_matrix[subset]:
                g_scores[subset,example] = g_scores[subset,example] + attribution_scores_tau[training_data_index,example]
    
    #MOF_scores = np.average(MOF_matrix,axis=-1)

    print("MOF Scores")
    print(MOF_matrix[1:,0])
    print("g_scores")
    print(g_scores[1:,0])

    from scipy.stats import spearmanr

    spearman_rank_scores = np.zeros((dataloader_X_size))
    for example in range(dataloader_X_size):
        spearman_rank_scores[example] = spearmanr(g_scores[1:,example],MOF_matrix[1:,example]).statistic
    
    print("spearman_rank_scores")
    print(spearman_rank_scores)

    return np.average(spearman_rank_scores,axis=-1)

    #return spearmanr(g_scores[1:],MOF_scores[1:])
#The original training code has a very annoying off-by-one-error, so we actually only have 31 subsets
#   Unrelated by s/o Matlab

#You ever make a devestating fuck up s.t. you have to spend hours writing this monstrosity?
# No?, me neither
def parseREADME(readme_path, sorted_index_map, num_subsets, ROWS_PER_CLASS_SUBSET, NUM_CLASSES):
    num_lines_subset_header = 5
    num_lines_file_header = 2
    subset_matrix = np.zeros((32,25000), np.int32)

    f = open(readme_path, "r")
    #Header
    for i in range(num_lines_file_header):
        f.readline()
    
    for subset_index in range(num_subsets):
        #Subset Header
        for i in range(num_lines_subset_header):
            f.readline()
        for class_index in range(NUM_CLASSES):
            for row_index in range(ROWS_PER_CLASS_SUBSET):
                l = f.readline()
                l = l.split("/")[-1]
                l = l.split(".")[0]
                img_file_index = int(l)
                subset_matrix[subset_index,class_index*ROWS_PER_CLASS_SUBSET+row_index] = sorted_index_map[class_index*ROWS_PER_CLASS_SUBSET+img_file_index]
    f.close()
    return subset_matrix

if __name__ == "__main__":
    LDS(None,None,None)