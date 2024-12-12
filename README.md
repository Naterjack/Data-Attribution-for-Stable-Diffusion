# Data Attribution for Stable Diffusion

## Requirements:
- Linux x64 OS, (we tested specifically with Ubuntu 24.04)
- A GPU with CUDA support and at least 24GB of VRAM
- At least 64GB of system memory

## Quickstart Guide

### Stage 0: Setup and aquire datasets
- Login to the Hugging Face CLI, (see this tutorial from them: https://huggingface.co/docs/huggingface_hub/en/guides/cli)

- Use environment.yml to create a conda environment SD3HF
```conda env create -f environemnt/environment_linux.yml```

- Run Generate_Local_CIFAR_10.ipynb to create a local copy of CIFAR10
- Run Generate_CIFAR_2.ipynb to create the CIFAR2 dataset

### Stage 1: Train a target model
- To train the model, you must first create a second conda environment specically for training
```conda env create -f environemnt/environment_training_linux.yml```
- Then you must obtain a local copy of HuggingFace's diffusers library using the following
```
conda activate SDTraining
cd ..
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
- In order to train on CIFAR10 (and CIFAR 2), you must edit the scripts in diffusers at
"```diffusers/examples/text_to_image/```". You need to add the following to both train_text_to_image_lora.py and train_text_to_image.py at lines 615, and 749 respectively (right under the comment '#We need to tokenize inputs and targets.')
```
    #print(dataset)
    cl = dataset["train"].features['label']
    #print(type(cl))

    def convertLabel(x):
        x['label_txt'] = cl.int2str(x['label'])
        return x

    dataset["train"] = dataset["train"].map(convertLabel)

    dataset = dataset.remove_columns(column_names=["label"])
```

- To fine tune a stable diffusion model, run the respective training script, for instance to LoRA fine tune on CIFAR10:
```
conda activate SDTraining
cd training_scripts/cifar10
./train_sd1_lora.sh
```

### Stage 2: Calculate and cache intermediaries for the attribution methods
```
conda activate SD3HF
```
- DINO
    - Run DINO_conversion.ipynb
- VAE
    - Run VAE_conversion.ipynb
- TRAK/D-TRAK
    - See TRAK/Train_TRAK.py, you _will_ need to make edits to the config section of this file, and depending on your choice of model type/training data, it may take a very long time to run, 30-100 hours is typical (hence why it is not provided as a notebook).

In all cases, the options for any config settings can be seen in utils/custom_enums.py .

### Stage 3: Calculate Attribution Scores
```
conda activate SD3HF
```
- DINO
    - Run SD1_DINO_Attribution.ipynb
- VAE
    - Run SD1_VAE_Attribution.ipynb
- TRAK/D-TRAK
    - Run SD1_TRAK_Attribution.ipynb

In all cases, make sure you match the configuration settings at the top of each file to that used in the previous stage.

### Stage 4: Evaluate Scores
- Class conformity
    - Class conformity results are generated at the end of the notebook in the previous stage.
- Counter factuals
    - Take the "config_name" printed at the end of the previous python notebook, and set it as the config_name in ```training_scripts/<dataset>/train_sd1_<model_type>_counterfactuals.sh```.
    - Train a counter factual model as follows:
```
conda activate SDTraining
cd training_scripts/<dataset>
./train_sd1_<model_type>_counterfactuals.sh
```
    - Once the counter factual model is trained, rerun the notebook from the previous stage with ```UPDATE_SCORES = False``` to generate a counter factual image.
    - Now run CounterFactualResults.ipynb to obtain image distances on the counter factual images generated.

### Other Notes:
- If you are on a Windows machine, you will need to change the default values of utils/config.py/project_config (i.e. IS_WINDOWS=True). Note that support for Windows is experimental and may break.
