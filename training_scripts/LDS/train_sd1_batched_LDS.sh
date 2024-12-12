export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export dataset_name="uoft-cs/cifar10"
export num_subsets=32

for i in `seq 9 $num_subsets`
do
accelerate launch --mixed_precision="fp16" ../diffusers/examples/text_to_image/train_text_to_image.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name="./datasets/cifar-10" \
--dataset_config_name="subset_$i" \
--use_ema \
--mixed_precision="fp16" \
--image_column="image" \
--caption_column="label_txt" \
--resolution=32 --random_flip \
--train_batch_size=128 \
--num_train_epochs=200 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--resume_from_checkpoint="latest" \
--checkpointing_steps=7800 \
--checkpoints_total_limit=1 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--output_dir="../../LDS/sd1-cifar10-v2/$i" \
--seed=$i
done