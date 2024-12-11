# AFAICT the only way to fix this is to make the whole project a module
#   Which is already annoying in theory and worse in this case since you cant have hypens in module names
import sys
import os
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from utils.config import CIFAR_10_Config, CIFAR_10_Local_Config, Project_Config, Dataset_Config
from utils.custom_enums import Dataset_Type_Enum, validate_enum
from typing import List
from pathlib import Path

from diffusers import DiffusionPipeline
import torch
import numpy as np

class Dataset_Generator(object):
    def __init__(
            self,
            project_config: Project_Config,
            dataset_type: Dataset_Type_Enum,
            dataset_config: Dataset_Config | None = None,
    ) -> None:
        
        self.project_config = project_config
        if dataset_config is not None:
            self.dataset_config = dataset_config
        else:
            if dataset_type == Dataset_Type_Enum.CIFAR10:
                self.dataset_config = CIFAR_10_Config()
            if dataset_type == Dataset_Type_Enum.CIFAR2:
                self.dataset_config = CIFAR_10_Local_Config(
                    project_config=project_config,
                    dataset_type=dataset_type
                )
        self.dataset_type = dataset_type

        #Tragically this doesn't save in the format we need
        #self.dataset_config.dataset.save_to_disk("./datasets/cifar-10/all")

        #Do NOT sort the dataset, since we need to preserve the original indicies
        #self.dataset_config.dataset = self.dataset_config.dataset.sort(column_names=self.dataset_config.caption_column)

        f = self.project_config.folder_symbol
        self.base_path = (
            project_config.PWD + f +
            "datasets" + f +
            self.dataset_type + f
        )

        self.base_path_images = (
            self.base_path + f +
            "train" + f
        )

    def generate_base_dataset(
            self,
            class_names_to_keep: List[str] | None = None,
            number_items_to_keep_per_class: int = 0,
    ):
        
        f = self.project_config.folder_symbol

        if class_names_to_keep is not None:
            #Check all provided classes are valid
            for class_name in class_names_to_keep:
                assert(class_name in self.dataset_config.class_captions)
            
            self.dataset_config.class_captions = class_names_to_keep
        
        if number_items_to_keep_per_class == 0:
            number_items_to_keep_per_class = len(self.dataset_config.dataset)

        number_items_per_class_tally = {}
        for class_caption in self.dataset_config.class_captions:
            number_items_per_class_tally[class_caption] = 0

        Path(self.base_path_images).mkdir(parents=True, exist_ok=True)

        for class_caption in self.dataset_config.class_captions:
            Path(self.base_path_images+class_caption+f).mkdir(parents=True, exist_ok=True)


        file_lines = []
        file_lines.append("---")
        file_lines.append("configs:")

        file_lines.append("  - config_name: default")
        file_lines.append("    drop_labels: false")
        file_lines.append("    data_files:")
        file_lines.append("      - split: train")
        file_lines.append("        path:")

        #In order to preserve the indicies, we have to make a README.md in the correct order, 
        # otherwise the dataset is loaded in class order

        #Note that there are two indicies here, i, which is the index in the original dataset
        #   and this value (j) which is the index in the new dataset
        j=0

        for i, item in enumerate(self.dataset_config.dataset):
            current_caption = item[self.dataset_config.caption_column]
            if current_caption in self.dataset_config.class_captions:
                if number_items_per_class_tally[current_caption] < number_items_to_keep_per_class:
                    number_items_per_class_tally[current_caption] += 1
                    item[self.dataset_config.image_column].save(f"{self.base_path_images}{item[self.dataset_config.caption_column]}{f}{j}.png")
                    file_lines.append(f"          - \"train{f}{item[self.dataset_config.caption_column]}{f}{j}.png\"")
                    j=j+1


        file_lines.append("---")

        for i in range(len(file_lines)):
            file_lines[i] = file_lines[i] + "\n"

        f = open(self.base_path+"README.md", "w")
        f.writelines(file_lines)
        f.close()
        return "Success!"

    def create_counterfactual_config(
            self,
            config_name:str, 
            indicies_to_remove:List[int]
    ) -> bool:
        
        f = self.project_config.folder_symbol
        indicies_to_remove.sort()
        current_index_to_remove = indicies_to_remove.pop(0)

        #Read the entire file so far
        file = open(self.base_path+"README.md", "r")
        file_lines = file.readlines()
        file.close()

        #Make sure that the config hasn't already been generated, 
        # as the code after this loop is potentially destructive in that case
        for line in file_lines:
            if len(line) > 2:
                if line[2] == '-':
                    if line.split(" ")[-1]==config_name:
                        print(f"Config {config_name} already exists!")
                        return False


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
                file_lines.append(f"          - \"train{f}{item[self.dataset_config.caption_column]}{f}{i}.png\"\n")
        file_lines.append("---\n")

        #Write the file, overwriting the whole thing
        # We do this rather than appending since we need to remove the last line
        file = open(self.base_path+"README.md", "w")
        file.writelines(file_lines)
        file.close()
        return True
