# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils.config import CIFAR_10_Config, Project_Config, Dataset_Config
from utils.custom_enums import Model_Type_Enum, Dataset_Type_Enum, validate_enum
from typing import List
from pathlib import Path

from diffusers import DiffusionPipeline
import torch
import numpy as np


class Counter_Factual_Image_Generator(object):
    def __init__(self,
                 project_config: Project_Config,
                 counter_factual_model_name: str,
                 model_type: Model_Type_Enum,
                 dataset_type: Dataset_Type_Enum
                 ) -> None:
        self.project_config = project_config
        self.model_type = validate_enum(model_type, Model_Type_Enum)
        self.dataset_type = validate_enum(dataset_type, Dataset_Type_Enum)
        self.counter_factual_model_name = counter_factual_model_name
    
    def generate_counter_factual_image(self,
                                       seed: int,
                                       image_class: str,
                                       save_image: bool,
                                       prompts: List[str]):
        f = self.project_config.folder_symbol
        p = (
            self.project_config.PWD + f + 
            "counter_factuals" + f + 
            self.dataset_type + f +
            self.counter_factual_model_name
        )
        assert(os.path.isdir(p))

        if self.model_type == Model_Type_Enum.LORA:
            #TODO: check this
            lora_weights_filepath = p + f + "pytorch_lora_weights.safetensors"
            pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
            if self.project_config.IS_CUDA:
                pipe.to("cuda")
            pipe.load_lora_weights(lora_weights_filepath)
        #if self.model_type == Model_Type_Enum.FULL:
        else:
            pipe = DiffusionPipeline.from_pretrained(p)
            if self.project_config.IS_CUDA:
                pipe.to("cuda")
        
        #This trick is directly based of something I found from DTRAK's code, see:
        #https://github.com/sail-sg/D-TRAK/blob/main/ArtBench5/train_text_to_image_lora.py 
        #line 1029 (at time of writing)
        #Notably, their version does not work as is in current versions of diffusers, 
        # and requires this slight adaption (creating a list of bools rather than a single bool)
        
        def dummy(images, **kwargs):
                return images, [False]*len(images)

        pipe.safety_checker = dummy

        #prompts=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck', 'frog']

        assert(image_class in prompts)

        #from here https://github.com/huggingface/diffusers/issues/1786#issuecomment-1359897983
        #   If this is in the proper documentation I sure couldn't find it easily
        #pipe.set_progress_bar_config(leave=False)
        pipe.set_progress_bar_config(disable=True)
        
        generator = torch.Generator(device=("cuda" if self.project_config.IS_CUDA else "cpu")).manual_seed(seed)

        image = pipe(
            height=32,
            width=32,
            prompt=image_class,
            generator=generator,
        ).images[0]

        if save_image:
            save_path = (p + f + "images")
            os.makedirs(save_path, exist_ok=True)
            image.save(save_path + f + image_class + "_" + str(seed)+ ".png")

        return image