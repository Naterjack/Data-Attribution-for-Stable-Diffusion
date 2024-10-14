import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from torchvision import transforms
import os
from trak import TRAKer
from SD1ModelOutput import SD1ModelOutput
from datasets import load_dataset
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

if IS_WINDOWS: 
    folder_symbol = "\\" 
else: #Assume UNIX-based
    folder_symbol = "/"

pwd = str(Path(__file__).resolve().parent.parent)
p = pwd + folder_symbol + MODEL_DIR + folder_symbol
assert(os.path.isdir(p))

ckpt_files = []
#s/o https://gist.github.com/madaan/6c9be9613e6760b7dee79bdfa621fc0f
for i in range(1,NUM_CHECKPOINTS+1):
    filename = (
                p + 
                "checkpoint-" + str(i*ITERATIONS_PER_CHECKPOINT) + folder_symbol + 
                "unet" + folder_symbol 
                + "diffusion_pytorch_model.bin"
                )
    if CONVERT_SAFETENSORS_TO_CKPT:
        ckpt_safetensors = load_file(filename.replace(".bin", ".safetensors"))
        torch.save(ckpt_safetensors, filename)
    ckpt_files.append(filename)
    #ckpt = torch.load(ckpt_filename)

ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

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

text_encoder.to("cuda")
vae.to("cuda")
unet.to("cuda")

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
#unet.train()

# Dataloader

#https://huggingface.co/datasets/

dataset_name = "uoft-cs/cifar10"
dataset = load_dataset(dataset_name, split="train")

image_column = "img"
caption_column = "label_txt"

#dataset = dataset.rename_column("img", "image")
print(dataset)
cl = dataset.features['label']
print(type(cl))

def convertLabel(x):
    x['label_txt'] = cl.int2str(x['label'])
    return x

dataset = dataset.map(convertLabel)

dataset = dataset.remove_columns(column_names=["label"])

train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(examples, is_train=True):
    inputs = tokenizer(
        examples[caption_column], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

#dataset = dataset.with_format("torch")
print(dataset)
train_dataset = dataset.with_transform(preprocess_train)

def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

# DataLoaders creation:
train_dataloader = DataLoader(
    train_dataset,
    shuffle=False,
    collate_fn=collate_fn,
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