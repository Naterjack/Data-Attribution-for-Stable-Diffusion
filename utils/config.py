from pathlib import Path
from os.path import isdir
from safetensors.torch import load_file
import torch
from typing import Iterable, List
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from datasets import load_dataset, Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.custom_enums import Model_Type_Enum, Dataset_Type_Enum, validate_enum

class Project_Config(object):
    def __init__(self,
                 PWD: str = "default",
                 IS_WINDOWS: bool = False,
                 IS_CUDA: bool = True,
                 ) -> None:
        self.IS_WINDOWS = IS_WINDOWS
        if self.IS_WINDOWS: 
            self.folder_symbol = "\\" 
        else: #Assume UNIX-based
            self.folder_symbol = "/"

        if PWD == "default":
            self.PWD = str(Path(__file__).resolve().parent.parent)
        else:
            self.PWD = PWD
        self.IS_CUDA = IS_CUDA;

class Model_Config(object):
    def __init__(self, 
                 project_config: Project_Config, 
                 MODEL_TYPE: Model_Type_Enum,
                 DATASET_TYPE: Dataset_Type_Enum,
                 NUM_CHECKPOINTS: int = 10,
                 ITERATIONS_PER_CHECKPOINT: int = 10000) -> None:
        self.project_config = project_config
        self.MODEL_TYPE = validate_enum(MODEL_TYPE, Model_Type_Enum)
        self.DATASET_TYPE = validate_enum(DATASET_TYPE, Dataset_Type_Enum)
        self.NUM_CHECKPOINTS = NUM_CHECKPOINTS
        self.ITERATIONS_PER_CHECKPOINT = ITERATIONS_PER_CHECKPOINT

    def getModelDirectory(self):
        f = self.project_config.folder_symbol
        p = (self.project_config.PWD +  f
             + "models" + f
             + self.DATASET_TYPE + f
             + "sd1-" + self.MODEL_TYPE + f)
        try:
            assert(isdir(p))
        except:
            raise Exception(f"Model directory {p} was not found!")
        return p
    
    def loadCheckpoints(self, 
                        p: str,
                        checkpoint_file_name: str = "diffusion_pytorch_model",
                        checkpoint_subfolder: str = "unet") -> Iterable[torch.Tensor]:
        ckpt_files = []
        #s/o https://gist.github.com/madaan/6c9be9613e6760b7dee79bdfa621fc0f
        for i in range(1, self.NUM_CHECKPOINTS+1):
            base_filename = (
                        p + 
                        "checkpoint-" + str(i*self.ITERATIONS_PER_CHECKPOINT) + self.project_config.folder_symbol + 
                        checkpoint_subfolder + self.project_config.folder_symbol 
                        + checkpoint_file_name
                        )
            bin_filename = base_filename + ".bin"
            safetensors_filename = base_filename +  ".safetensors"
            if Path(safetensors_filename).is_file() and not(Path(bin_filename).is_file()):
                ckpt_safetensors = load_file(safetensors_filename)
                torch.save(ckpt_safetensors, bin_filename)
            ckpt_files.append(bin_filename)
            #ckpt = torch.load(ckpt_filename)
        
        ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]
        return ckpts
    
    def loadModelComponents(self, 
                            p: str) -> tuple[CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel]:
        tokenizer = CLIPTokenizer.from_pretrained(
            p, subfolder="tokenizer",
        )
        text_encoder = CLIPTextModel.from_pretrained(
            p, subfolder="text_encoder",
        )
        vae = AutoencoderKL.from_pretrained(
            p, subfolder="vae",
        )
        unet = UNet2DConditionModel.from_pretrained(
            p, subfolder="unet",
        )
        return tokenizer, text_encoder, vae, unet

class LoRA_Model_Config(Model_Config):

    def loadLoRAUnet(self,
                     p: str,
                     checkpoint_index: int,
                     base_unet_huggingface_slug: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                     checkpoint_file_name: str = "pytorch_lora_weights.safetensors"
                     ) -> UNet2DConditionModel:
        unet = UNet2DConditionModel.from_pretrained(
            base_unet_huggingface_slug, subfolder="unet",
        )
        ### Again the following is based on train_text_to_image_lora.py
        unet.requires_grad_(False)
        # Freeze the unet parameters before adding adapters
        for param in unet.parameters():
            param.requires_grad_(False)

        lora_file = (
                    p + 
                    "checkpoint-" + str((checkpoint_index+1)*self.ITERATIONS_PER_CHECKPOINT) +
                    self.project_config.folder_symbol +
                    checkpoint_file_name
                    )
        unet.load_attn_procs(lora_file)
        return unet
        
class Dataset_Config(object):
    def __init__(self, 
                 huggingface_slug: str,
                 config_name: str | None = None,
                 ) -> None:
        self.huggingface_slug = huggingface_slug
        self.dataset_config_name = config_name
        if self.dataset_config_name is not None:
            self.dataset = load_dataset(self.huggingface_slug, name=self.dataset_config_name, split="train")
        else:
            self.dataset = load_dataset(self.huggingface_slug, split="train")

class CIFAR_10_Config(Dataset_Config):
    def __init__(self, 
                 huggingface_slug: str = "uoft-cs/cifar10",
                 config_name: str | None = None,
                 existing_image_column_name: str = "img", 
                 existing_caption_column_name: str = "label",
                 new_image_column_name: str = "image", 
                 new_caption_column_name: str = "label_txt",
                 ) -> None:
        super().__init__(huggingface_slug, config_name)
        # Dataloader
        #https://huggingface.co/datasets/
        self.image_column = new_image_column_name
        self.caption_column = new_caption_column_name

        if existing_image_column_name != new_image_column_name:
            self.dataset = self.dataset.rename_column(existing_image_column_name, new_image_column_name)

        cl = self.dataset.features[existing_caption_column_name]

        self.dataset = self.dataset.map(lambda x, cIn, cOut: {cOut: cl.int2str(x[cIn])}, 
                                        fn_kwargs={
                                            "cIn": existing_caption_column_name, 
                                            "cOut": new_caption_column_name
                                            }
                                        )
        
        self.class_captions = cl.names
        
        self.dataset = self.dataset.remove_columns(column_names=[existing_caption_column_name])

        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __setTokenizer(self, tokenizer: CLIPTokenizer | None):
        #Okay so I _think_ this will work to trick GC into doing what I want
        self.tokenizer = tokenizer
    
    def __destroyTokenizer(self):
        self.__setTokenizer(None)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def __tokenize_captions(self, examples, is_train=True):
        inputs = self.tokenizer(
            examples[self.caption_column], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    #TODO: Convert this to private
    def preprocess_train(self, examples):
        images = [image.convert("RGB") for image in examples[self.image_column]]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self.__tokenize_captions(examples)
        return examples

    def preprocess(self, tokenizer: CLIPTokenizer):
        self.__setTokenizer(tokenizer)
        train_dataset = self.dataset.with_transform(self.preprocess_train)
        #So ".with_transform" seems to be a JIT based vibe rather than being done ahead of time
        # This is kinda tragic, and it means we can't simply destroy the tokenzier once we're done
        #self.__destroyTokenizer()
        return train_dataset

    def collate_fn(self, examples): #This might be a breaking change, we will see
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

class CIFAR_10_Local_Config(CIFAR_10_Config):
    def __init__(self,
                 project_config: Project_Config,
                 dataset_type: Dataset_Type_Enum,
                 #huggingface_slug: str = "uoft-cs/cifar10", 
                 config_name: str | None = None,
                 existing_image_column_name: str = "image", 
                 existing_caption_column_name: str = "label", 
                 new_image_column_name: str = "image", 
                 new_caption_column_name: str = "label_txt") -> None:
        dataset_type = validate_enum(dataset_type, Dataset_Type_Enum)
        f =  project_config.folder_symbol
        super().__init__(f"{project_config.PWD}{f}datasets{f}{dataset_type}", 
                         config_name, 
                         existing_image_column_name, 
                         existing_caption_column_name, 
                         new_image_column_name, 
                         new_caption_column_name)

if __name__ == "__main__":
    c10 = CIFAR_10_Config()