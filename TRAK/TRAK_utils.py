from os import makedirs
import torch
import numpy as np
from numpy.lib.format import open_memmap

#Python actually high key stinks
# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils.config import Project_Config, LoRA_Model_Config, Model_Config
from utils.custom_enums import TRAK_Type_Enum, Dataset_Type_Enum, Model_Type_Enum, validate_enum
from trak import TRAKer, projectors
from TRAK.SD1ModelOutput import SD1ModelOutput

class TRAK_Config(object):
    def __init__(self,
                 project_config: Project_Config,
                 model_type: Model_Type_Enum,
                 TRAK_type: TRAK_Type_Enum,
                 dataset_type: Dataset_Type_Enum,
                 ) -> None:
        self.project_config = project_config
        self.dataset_type = validate_enum(dataset_type, Dataset_Type_Enum)
        self.TRAK_type = validate_enum(TRAK_type, TRAK_Type_Enum)
        self.model_type = validate_enum(model_type, Model_Type_Enum)

        self.MODEL_NAME_CLEAN = f"sd1-{model_type}"
        #model_dir = f"sd1-{model_type}-{dataset_type}"

        if dataset_type == Dataset_Type_Enum.CIFAR10:
            iterations_per_checkpoint=10000
        if dataset_type == Dataset_Type_Enum.CIFAR2:
            iterations_per_checkpoint=1000

        if model_type == Model_Type_Enum.LORA:
            self.model_config = LoRA_Model_Config(
                project_config=project_config,
                MODEL_TYPE=model_type,
                DATASET_TYPE=dataset_type,
                NUM_CHECKPOINTS=10,
                ITERATIONS_PER_CHECKPOINT=iterations_per_checkpoint,
            )
        else:
            self.model_config = Model_Config(
                project_config=project_config,
                MODEL_TYPE=model_type,
                DATASET_TYPE=dataset_type,
                NUM_CHECKPOINTS=10,
                ITERATIONS_PER_CHECKPOINT=iterations_per_checkpoint,
            )
        
        f = project_config.folder_symbol
        self.TRAK_SAVE_DIR = f"{project_config.PWD}{f}TRAK{f}results{f}{self.dataset_type}{f}{self.TRAK_type}_{self.MODEL_NAME_CLEAN}"
        #TODO This doesnt do what I want it to
        makedirs(self.TRAK_SAVE_DIR, exist_ok=True)
        print(f"TRAK is being saved to {self.TRAK_SAVE_DIR}")

    #https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    def __count_parameters(self, model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def load_checkpoints(self):
        p = self.model_config.getModelDirectory()
        if self.model_type == Model_Type_Enum.LORA:
            tokenizer, text_encoder, vae, _ = self.model_config.loadModelComponents("stable-diffusion-v1-5/stable-diffusion-v1-5")
            unet = self.model_config.loadLoRAUnet(p,0)
            ckpts = range(self.model_config.NUM_CHECKPOINTS)
        else:
            ckpts = self.model_config.loadCheckpoints(p)
            tokenizer, text_encoder, vae, unet = self.model_config.loadModelComponents(p)
        
        return tokenizer, text_encoder, vae, unet, ckpts

    def load_model_into_TRAKer(self, 
                    model,
                    train_set_size):
        num_params = self.__count_parameters(model)
        print(f"Model contains {num_params} trainable parameters")
        if self.model_type == Model_Type_Enum.LORA:
            lora_layers = []
            lora_layers_filter = filter(lambda p: p[1].requires_grad, model.named_parameters())
            for layer in lora_layers_filter:
                lora_layers.append(layer[0])
            
            #This works around a bug in TRAKer which counts ALL parameters (including those which aren't trainable)
            #   when deciding which projector to use. 
            # We can simply specifiy a projector manually to get around this.
            projector = projectors.CudaProjector(grad_dim=num_params,
                                    proj_dim=1024,
                                    seed=0,
                                    proj_type=projectors.ProjectionType.rademacher,
                                    max_batch_size=8,
                                    dtype=torch.float16,
                                    device="cuda")
        else:
            lora_layers = None
            projector = None

        model_output = SD1ModelOutput(
            project_config=self.project_config,
            TRAK_type=self.TRAK_type
        )
        #MODEL_DIR=self.model_config.MODEL_DIR) #Leave this hardcoded since
        #   its the same in all instances
        
        traker = TRAKer(model=model,
                        task=model_output,
                        train_set_size=train_set_size, #len(train_dataloader.dataset),
                        save_dir=self.TRAK_SAVE_DIR,
                        proj_max_batch_size=8, #Default 32, requires an A100 apparently
                        proj_dim=1024,
                        grad_wrt=lora_layers,
                        projector=projector,
                        )
        return traker
     
    def load_TRAKer(self, train_set_size):
        tokenizer, text_encoder, vae, unet, ckpts = self.load_checkpoints()
        
        if self.project_config.IS_CUDA:
            text_encoder.to("cuda")
            vae.to("cuda")
            unet.to("cuda")

        # Freeze vae and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        traker = self.load_model_into_TRAKer(unet,train_set_size)
        return traker, tokenizer, text_encoder, vae, unet, ckpts

class TRAK_Experiment_Config(TRAK_Config):
    def __init__(self, 
                 project_config: Project_Config,
                 model_type: Model_Type_Enum,
                 TRAK_type: TRAK_Type_Enum,
                 dataset_type: Dataset_Type_Enum,
                 FORCE_FULL_MODEL_TEST_DATASET: bool
                 ) -> None:
        #FORCE_FULL_MODEL_TEST_DATASET allows for taking a TRAKer instance that 
        # has been featurized on on a LoRA model, and getting scoring it on
        # a "full" model.
        super().__init__(project_config, model_type, TRAK_type, dataset_type)
        self.FORCE_FULL_MODEL_TEST_DATASET = FORCE_FULL_MODEL_TEST_DATASET

        if self.FORCE_FULL_MODEL_TEST_DATASET and (self.model_type == Model_Type_Enum.FULL):
            raise ValueError("Cannot override the test dataset for a Non-LoRA model")
        
        if self.model_type == Model_Type_Enum.LORA:
            if FORCE_FULL_MODEL_TEST_DATASET:
                EXPERIMENT_SUBTITLE = "_full_model_test_dataset"
            else:
                EXPERIMENT_SUBTITLE = "_normal_test_dataset"
                
            self.EXPERIMENT_NAME = self.MODEL_NAME_CLEAN+EXPERIMENT_SUBTITLE
        else:
            self.EXPERIMENT_NAME = self.MODEL_NAME_CLEAN
        
        if TRAK_type==TRAK_Type_Enum.TRAK and (self.model_type == Model_Type_Enum.FULL):
            #if UPDATE_SCORES:
            print("Warning, this particular combination is expensive to compute.")
            print("On our 3090 system, this requires 23.95GB of VRAM (97%), and takes 25 min to compute scores")
    
    def load_scores(self):
        f = self.project_config.folder_symbol
        memmap_path = ( 
                    self.TRAK_SAVE_DIR + 
                    f +
                    "scores" +
                    f +
                    self.EXPERIMENT_NAME +
                    ".mmap")
        scores_org = open_memmap(memmap_path, mode='c')
        scores = np.copy(scores_org)
        return scores
