REM export MODEL_NAME="runwayml/stable-diffusion-v1-5"
REM export dataset_name="uoft-cs/cifar10"

accelerate launch train_text_to_image.py ^
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ^
  --dataset_name="uoft-cs/cifar10" ^
  --use_ema ^
  --resolution=32 --random_flip ^
  --train_batch_size=1 ^
  --gradient_accumulation_steps=4 ^
  --gradient_checkpointing ^
  --mixed_precision="fp16" ^
  --max_train_steps=15000 ^
  --learning_rate=1e-05 ^
  --max_grad_norm=1 ^
  --lr_scheduler="constant" --lr_warmup_steps=0 ^
  --output_dir="sd1-cifar10" 