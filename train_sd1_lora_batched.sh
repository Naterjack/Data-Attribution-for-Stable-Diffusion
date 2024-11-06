export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export dataset_name="uoft-cs/cifar10"

accelerate launch --mixed_precision="fp16" ../diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --dataloader_num_workers=8 \
  --mixed_precision="fp16" \
  --image_column="img" \
  --caption_column="label_txt" \
  --resolution=32 --random_flip \
  --train_batch_size=128 \
  --num_train_epochs=200 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --resume_from_checkpoint="latest" \
  --checkpointing_steps=7800 \
  --checkpoints_total_limit=10 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="sd1-cifar10-v2-lora-batched-unstable-v2" \
  --seed=42