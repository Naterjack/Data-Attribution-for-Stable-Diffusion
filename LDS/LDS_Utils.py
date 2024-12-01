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

from utils.config import CIFAR_10_Config, Project_Config, Model_Config
from utils.custom_enums import Dataset_Type_Enum, Model_Type_Enum

NUM_TIMESTEPS_TO_SAMPLE = 100

'''

'''
def LDS(project_config: Project_Config,
        training_dataset_config: CIFAR_10_Config,
        generated_dataset_dataloader: DataLoader,
        training_dataset_type: Dataset_Type_Enum,
        generated_dataset_size: int,
        attribution_scores_tau: np.array, 
        seed: int,
        training_subsets_count: int = 31, #sshhhhhhh
        training_subsets_size: float = 0.5,
        ):

    NUM_SUBSETS = training_subsets_count
    TRAINING_DATASET_LENGTH = training_dataset_config.dataset.num_rows
    GENERATED_DATASET_LENGTH = generated_dataset_size

    NUM_CLASSES = len(training_dataset_config.class_captions)

    SUBSET_SIZE_ALPHA = training_subsets_size
    SUBSET_LENGTH = int(TRAINING_DATASET_LENGTH * SUBSET_SIZE_ALPHA)

    subsets_base_path = (project_config.PWD + project_config.folder_symbol +
                "datasets" + project_config.folder_symbol +
                training_dataset_type + project_config.folder_symbol)
    
    subset_matrix = parseREADME(
        readme_path=subsets_base_path+"LDS_CONFIG.md",
        NUM_SUBSETS=NUM_SUBSETS,
        SUBSET_LENGTH=SUBSET_LENGTH
    )

    DATASET_NAME = f"sd1-{training_dataset_type}"

    models_base_path = (project_config.PWD + project_config.folder_symbol +
                "LDS" + project_config.folder_symbol +
                DATASET_NAME + project_config.folder_symbol)

    MOF_matrix = np.zeros((GENERATED_DATASET_LENGTH,NUM_SUBSETS))

    generator = torch.Generator(device=("cuda" if project_config.IS_CUDA else "cpu")).manual_seed(seed)
    
    cache_file_name = "LDS_CIFAR10_MOF_CACHE"
    try:
        if os.path.isfile(project_config.PWD + project_config.folder_symbol + cache_file_name + ".npy"):
            MOF_matrix = np.load(cache_file_name+".npy")
        else:
            for i in tqdm(range(NUM_SUBSETS)):
                ### Load Model
                model_path = models_base_path + str(i+1)
                model_config = Model_Config(
                    project_config=project_config,
                    MODEL_TYPE=Model_Type_Enum.FULL,
                    DATASET_TYPE=training_dataset_type,
                )
                _, text_encoder, vae, unet = model_config.loadModelComponents(model_path)
                if project_config.IS_CUDA:
                    text_encoder.to("cuda")
                    vae.to("cuda")
                    unet.to("cuda")
                
                batch_counter=0
                for batch in generated_dataset_dataloader:
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
                    latents = latents.repeat(NUM_TIMESTEPS_TO_SAMPLE,1,1,1)
                    encoder_hidden_states = encoder_hidden_states.repeat(NUM_TIMESTEPS_TO_SAMPLE,1,1)
                    #This code is (mostly) taken from the original huggingface training code, since their API
                    # does not let you get the loss directly
                    # The original code can be found here: https://github.com/huggingface/diffusers/blob/v0.30.3/examples/text_to_image/train_text_to_image.py

                    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    if NUM_TIMESTEPS_TO_SAMPLE == 1:
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, 
                                                noise_scheduler.config.num_train_timesteps, 
                                                (bsz,),
                                                generator=generator,
                                                device=latents.device)
                    else:
                        timesteps = torch.arange(0 if NUM_TIMESTEPS_TO_SAMPLE==1000 else 1, 
                                                noise_scheduler.config.num_train_timesteps,
                                                noise_scheduler.config.num_train_timesteps//NUM_TIMESTEPS_TO_SAMPLE, 
                                                dtype=torch.int64,
                                                device=latents.device)
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
                    #target = target.unsqueeze(0)

                    loss = F.mse_loss(target, model_pred[0])

                    loss = loss.detach().cpu()

                    #print(MOF_matrix.shape)
                    #print(i)
                    #print(batch_counter)

                    MOF_matrix[batch_counter,i] = loss
                    batch_counter += 1
            np.save(cache_file_name, MOF_matrix)
    except:
        raise IOError("Something went wrong attempting to read or discover cache: Did you delete a file?") 

    g_scores = np.zeros((GENERATED_DATASET_LENGTH, NUM_SUBSETS))
    for generated_example_index in range(GENERATED_DATASET_LENGTH):
        for subset in range(NUM_SUBSETS):
            for training_data_index in subset_matrix[subset]:
                g_scores[generated_example_index,subset] = g_scores[generated_example_index,subset] + attribution_scores_tau[training_data_index,generated_example_index]
    
    #MOF_scores = np.average(MOF_matrix,axis=-1)

    print("MOF Scores")
    print(MOF_matrix[0,:])
    print("g_scores")
    print(g_scores[0,:])

    from scipy.stats import spearmanr

    spearman_rank_scores = np.zeros((GENERATED_DATASET_LENGTH))
    for generated_example_index in range(GENERATED_DATASET_LENGTH):
        spearman_rank_scores[generated_example_index] = spearmanr(g_scores[generated_example_index],MOF_matrix[generated_example_index]).statistic
    
    print("spearman_rank_scores")
    print(spearman_rank_scores)

    return np.average(spearman_rank_scores,axis=-1)

    #return spearmanr(g_scores[1:],MOF_scores[1:])
    #The original training code has a very annoying off-by-one-error, so we actually only have 31 subsets
    #   Unrelated by s/o Matlab

    #You ever make a devestating fuck up s.t. you have to spend hours writing this monstrosity?
    # No?, me neither
def parseREADME_scuffed(readme_path, sorted_index_map, num_subsets, ROWS_PER_CLASS_SUBSET, NUM_CLASSES):
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

def parseREADME(readme_path, NUM_SUBSETS, SUBSET_LENGTH):
    num_lines_subset_header = 5
    num_lines_file_header = 2

    subset_matrix = np.zeros((NUM_SUBSETS,SUBSET_LENGTH), np.int32)

    f = open(readme_path, "r")
    #Header
    for i in range(num_lines_file_header):
        f.readline()
    
    for subset_index in range(NUM_SUBSETS):
        for i in range(num_lines_subset_header):
                f.readline()
        for i in range(SUBSET_LENGTH):
            l = f.readline()
            l = l.split("/")[-1]
            l = l.split(".")[0]
            img_index = int(l)
            subset_matrix[subset_index,i] = img_index

    f.close()
    return subset_matrix

if __name__ == "__main__":
    LDS(None,None,None)