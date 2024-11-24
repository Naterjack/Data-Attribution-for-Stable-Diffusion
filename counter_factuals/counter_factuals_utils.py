#Python actually high key stinks
# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils.config import CIFAR_10_Config, Project_Config
from typing import List
from pathlib import Path

from diffusers import DiffusionPipeline
import torch

class Counter_Factual_Dataset_Generator(object):
    def __init__(self) -> None:
        self.DATASET_NAME = "cifar-10-original-indicies"
        self.dataset_config = CIFAR_10_Config()
        self.project_config = Project_Config()
        #Tragically this doesn't save in the format we need
        #self.dataset_config.dataset.save_to_disk("./datasets/cifar-10/all")

        #Do NOT sort the dataset, since we need to preserve the original indicies
        #self.dataset_config.dataset = self.dataset_config.dataset.sort(column_names=self.dataset_config.caption_column)

        self.base_path = ("." + self.project_config.folder_symbol +
                    "datasets" + self.project_config.folder_symbol +
                    self.DATASET_NAME + self.project_config.folder_symbol)

        self.base_path_images = (self.base_path + self.project_config.folder_symbol +
                            "train" + self.project_config.folder_symbol)

    def generate_base_dataset(self):
        Path(self.base_path_images).mkdir(parents=True, exist_ok=True)

        for class_caption in self.dataset_config.class_captions:
            Path(self.base_path_images+class_caption+self.project_config.folder_symbol).mkdir(parents=True, exist_ok=True)


        file_lines = []
        file_lines.append("---")
        file_lines.append("configs:")

        file_lines.append("  - config_name: full")
        file_lines.append("    drop_labels: false")
        file_lines.append("    data_files:")
        file_lines.append("      - split: train")
        file_lines.append("        path:")

        #In order to preserve the indicies, we have to make a README.md in the correct order, 
        # otherwise the dataset is loaded in class order


        for i, item in enumerate(self.dataset_config.dataset):
            item[self.dataset_config.image_column].save(f"{self.base_path_images}{item[self.dataset_config.caption_column]}{self.project_config.folder_symbol}{i}.png")
            file_lines.append(f"          - \"train{self.project_config.folder_symbol}{item[self.dataset_config.caption_column]}{self.project_config.folder_symbol}{i}.png\"")


        file_lines.append("---")

        for i in range(len(file_lines)):
            file_lines[i] = file_lines[i] + "\n"

        f = open(self.base_path+"README.md", "w")
        f.writelines(file_lines)
        f.close()
        return "Success!"

    def create_counterfactual_config(self,
                                     config_name:str, 
                                     indicies_to_remove:List[int]):
        indicies_to_remove.sort()
        current_index_to_remove = indicies_to_remove.pop(0)

        #Read the entire file so far
        f = open(self.base_path+"README.md", "r")
        file_lines = f.readlines()
        f.close()

        #Delete last line ("---")
        file_lines.pop()

        #Add our new config
        file_lines.append(f"  - config_name: {config_name}\n")
        file_lines.append("    drop_labels: false\n")
        file_lines.append("    data_files:\n")
        file_lines.append("      - split: train\n")
        file_lines.append("        path:\n")
        for i, item in enumerate(self.dataset_config.dataset):
            if i == current_index_to_remove:
                if len(indicies_to_remove) > 0: 
                    current_index_to_remove = indicies_to_remove.pop(0)
            else:
                file_lines.append(f"          - \"train{self.project_config.folder_symbol}{item[self.dataset_config.caption_column]}{self.project_config.folder_symbol}{i}.png\"\n")
        file_lines.append("---\n")

        #Write the file, overwriting the whole thing
        # We do this rather than appending since we need to remove the last line
        f = open(self.base_path+"README.md", "w")
        f.writelines(file_lines)
        f.close()

class Counter_Factual_Image_Generator(object):
    def __init__(self,
                 project_config: Project_Config,
                 counter_factual_model_name: str,
                 is_lora: bool,
                 ) -> None:
        self.project_config = project_config
        self.IS_LORA = is_lora
        self.counter_factual_model_name = counter_factual_model_name
    
    def generate_counter_factual_image(self,
                                       seed: int,
                                       image_class: str,
                                       save_image: bool):
        f = self.project_config.folder_symbol
        p = self.project_config.PWD + f + "counter_factuals" + f + self.counter_factual_model_name
        assert(os.path.isdir(p))

        if self.IS_LORA:
            #TODO: check this
            p = p + f + "pytorch_lora_weights.safetensors"
            pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
            if self.project_config.IS_CUDA:
                pipe.to("cuda")
            pipe.load_lora_weights(p)
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

        prompts=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck', 'frog']

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