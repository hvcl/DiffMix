"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision as tv
from PIL import Image

from guided_diffusion_sample.image_datasets_pathology import load_data
from guided_diffusion_sample import dist_util
from guided_diffusion_sample.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    
    args.idx_img = args.idx_img.split(',')
    args.results_path += f'/s{args.s}_{args.timestep_respacing}_p{args.ddim_percent}'

    ddim_percent = args.ddim_percent * 0.01

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    dist_util.setup_dist()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    print("creating data loader...")
    data_loader = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        idx_img=args.idx_img
    )
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    sample_path = os.path.join(args.results_path, 'samples')
    npy_patch_path = os.path.join(args.results_path, 'npy_patch')

    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(npy_patch_path, exist_ok=True)

    print("sampling...")
    all_samples = []
    n_processed_patches = 0
    for i, (batch, cond) in enumerate(data_loader):
        patch_indices = cond['patch_idx'].numpy()
        type_maps = cond['label'][:,0,:,:].numpy().astype(np.uint8)
        inst_maps = cond["instance"][:,0,:,:].numpy().astype(np.int32)
        model_kwargs = preprocess_input(cond, num_classes=args.num_classes)
        
        # set hyperparameter
        model_kwargs['gt'] = batch.cuda()
        model_kwargs['s'] = args.s

        sample_fn = diffusion.ddim_sample_loop
        sample = sample_fn(
            model,
            (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            ddim_percent=ddim_percent
        )
        sample = (sample + 1) / 2.0

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(sample.shape[0]):
            patch_idx = patch_indices[j]
            patch_fn = cond['path'][j].split('/')[-1].split('.')[0] + f'_{patch_idx:03d}'
            tv.utils.save_image(sample[j], os.path.join(sample_path, patch_fn + '.png'))
            
            patch_pth = os.path.join(npy_patch_path, patch_fn + '.npy')
            
            cur_img  = sample[j].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", th.uint8).numpy()
            cur_inst = inst_maps[j].astype(np.int32)
            cur_type = type_maps[j].astype(np.uint8)
            
            inst_map = np.expand_dims(cur_inst, axis=-1)
            type_map = np.expand_dims(cur_type, axis=-1)
            patch_npy = np.concatenate([cur_img, inst_map, type_map], axis=-1) # [256, 256, 5]
            
            np.save(file=patch_pth, arr=patch_npy)

        n_processed_patches += batch.shape[0]

        print(f"Created {n_processed_patches} samples")

    dist.barrier()
    print("sampling complete")


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        s=1.0,
        gpu="0",
        ddim_percent=100,
        idx_img='0',
        timestep_respacing='ddim100'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()