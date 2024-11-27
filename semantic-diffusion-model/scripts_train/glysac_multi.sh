# Training
export OPENAI_LOGDIR='OUTPUT/glysac-SDM-256CH'
# mpiexec -n 8 
python image_train.py \
--data_dir /Dataset/glysac \
--dataset_mode glysac \
--lr 1e-4 \
--batch_size 16 \
--attention_resolutions 32,16,8 \
--diffusion_steps 1000 \
--image_size 256 \
--learn_sigma True \
--noise_schedule linear \
--num_channels 256 \
--num_head_channels 64 \
--num_res_blocks 2 \
--resblock_updown True \
--use_fp16 True \
--use_scale_shift_norm True \
--use_checkpoint True \
--num_classes 4 \
--class_cond True \
--no_instance False \
--save_interval 120 \
--gpu 0

# Classifier-free Finetune
export OPENAI_LOGDIR='OUTPUT/glysac-SDM-256CH-FINETUNE'
python image_train.py \
--data_dir /Dataset/glysac \
--dataset_mode glysac \
--lr 2e-5 \
--batch_size 16 \
--attention_resolutions 32,16,8 \
--diffusion_steps 1000 \
--image_size 256 \
--learn_sigma True \
--noise_schedule linear \
--num_channels 256 \
--num_head_channels 64 \
--num_res_blocks 2 \
--resblock_updown True \
--use_fp16 True \
--use_scale_shift_norm True \
--use_checkpoint True \
--num_classes 4 \
--class_cond True \
--drop_rate 0.2 \
--no_instance False \
--resume_checkpoint OUTPUT/glysac-SDM-256CH-FINETUNE/model009480.pt \
--save_interval 120 \
--gpu 1

# Testing (orig SDM)
export OPENAI_LOGDIR='OUTPUT/glysac-SDM-256CH-TEST'
python image_sample.py \
--data_dir /Dataset/glysac \
--dataset_mode glysac \
--attention_resolutions 32,16,8 \
--diffusion_steps 1000 \
--image_size 256 \
--learn_sigma True \
--noise_schedule linear \
--num_channels 256 \
--num_head_channels 64 \
--num_res_blocks 2 \
--resblock_updown True \
--use_fp16 True \
--use_scale_shift_norm True \
--num_classes 4 \
--class_cond True \
--batch_size 2 \
--num_samples 3 \
--no_instance False \
--model_path OUTPUT/glysac-SDM-256CH-FINETUNE/ema_0.9999_010200.pt \
--results_path RESULTS/glysac-SDM-256CH \
--s 2.0 \
--use_ddim \
--gpu 2
