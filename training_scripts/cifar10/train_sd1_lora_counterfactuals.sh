export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export dataset_name="../../datasets/cifar10"
export config_name="VAE_sd1-lora_frog_seed_1"

accelerate launch --mixed_precision="fp16" ../../../diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --dataset_config_name=$config_name \
  --mixed_precision="fp16" \
  --image_column="image" \
  --caption_column="label_txt" \
  --resolution=32 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --resume_from_checkpoint="latest" \
  --checkpointing_steps=10000 \
  --checkpoints_total_limit=10 \
  --max_train_steps=100000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="../../counter_factuals/cifar10/lora/$config_name" \
  --seed=42