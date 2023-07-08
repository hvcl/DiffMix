python image_sample.py \
--data_dir /Dataset/glysac/Train \
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
--no_instance False \
--model_path OUTPUT/glysac-SDM-256CH-FINETUNE/ema_0.9999_010200.pt \
--results_path RESULTS/glysac-grad_sdm \
--timestep_respacing ddim100 \
--s 1.5 \
--batch_size 80 --ddim_percent 55 --gpu 0 --idx_img 0