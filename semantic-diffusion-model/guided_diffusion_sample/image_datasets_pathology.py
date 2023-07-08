import os
import math
import random

from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import measurements


def load_data(
    *,
    dataset_mode, # glysac
    data_dir, # /Dataset/glysac
    batch_size, # 4
    image_size, # 256
    class_cond=False, # True
    idx_img=0
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset_mode in ['glysac', 'consep']:
        all_img_paths = _list_image_files_recursively(os.path.join(data_dir, 'Images'))
        all_mask_paths = _list_image_files_recursively(os.path.join(data_dir, 'Labels'))
        target_img_paths, target_mask_paths = list(), list()
        for idx in idx_img:
            target_img_paths.append(all_img_paths[int(idx)])
            target_mask_paths.append(all_mask_paths[int(idx)])
    else:
        raise NotImplementedError('{} not implemented'.format(dataset_mode))

    print("Len of Dataset:", len(target_img_paths))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        target_img_paths,
        target_mask_paths,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False
    )

    return loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        patch_size,
        image_paths,
        mask_paths
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_paths = image_paths
        path_target_mask = f'/Dataset/{dataset_mode}/Train/Gradmix_Labels'

        self.imgs, self.ann_orig_type, self.ann_type, self.ann_inst = list(), list(), list(), list()

        # data preprocessing
        for image_path, mask_path in zip(image_paths, mask_paths):
            img = Image.open(image_path)
            img = np.array(img.convert("RGB"))
            img = img.transpose([2, 0, 1])
            
            fn_target = os.path.basename(mask_path).split('.')[0] + '_synthesized.mat'
            path_target = os.path.join(path_target_mask, fn_target)
            ann = sio.loadmat(path_target)
            ann_inst = np.expand_dims(ann["inst_map"], axis=0).astype(np.int32)
            ann_type = np.expand_dims(ann["type_map"], axis=0).astype(np.uint8)

            ann_type = label_type_map(dataset_mode, ann_type)

            self.imgs.append(img)
            self.ann_inst.append(ann_inst)
            self.ann_type.append(ann_type)

        self.n_patches = 100

    def __len__(self):
        return len(self.imgs)*self.n_patches

    def __getitem__(self, idx):
        np.random.seed()

        idx_p = int(np.mod(idx, self.n_patches))
        index = int(np.floor(idx/self.n_patches))
        
        img_path = self.img_paths[index]
        img = self.imgs[index]
        
        inst_map = self.ann_inst[index]
        type_map = self.ann_type[index]

        Y, X = H[idx_p], W[idx_p]
        img = img[:, Y:Y+self.patch_size, X:X+self.patch_size]
        inst_map = inst_map[:, Y:Y+self.patch_size, X:X+self.patch_size]
        type_map = type_map[:, Y:Y+self.patch_size, X:X+self.patch_size]

        img = img.astype(np.float32) / 127.5 - 1
        img = np.clip(img, -1, 1)
        img = img.copy()

        inst_map = remap_label(inst_map)
        type_map = type_map.astype(np.uint8)

        type_map_orig = type_map.copy()
        
        template_inst = inst_map.copy()
        template_type = type_map.copy()
            
        out_dict = {
            "path": img_path,
            "label": template_type.copy(),
            "instance": template_inst.copy(),
            "label_ori": type_map_orig,
            "patch_idx": idx_p
        }
        
        return img, out_dict

def remap_label(mask, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        mask    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    mask_id = list(np.unique(mask))
    mask_id.remove(0)
    if len(mask_id) == 0:
        return mask  # no label
    if by_size:
        mask_size = []
        for inst_id in mask_id:
            size = (mask == inst_id).sum()
            mask_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(mask_id, mask_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        mask_id, mask_size = zip(*pair_list)

    new_mask = np.zeros(mask.shape, np.int32)
    for idx, inst_id in enumerate(mask_id):
        new_mask[mask == inst_id] = idx + 1
    return new_mask


def label_type_map(dataset, type_map):
    if dataset == "consep":
        type_map[(type_map == 3) | (type_map == 4)] = 3
        type_map[(type_map == 5) | (type_map == 6) | (type_map == 7)] = 4
    elif dataset == "glysac":
        type_map[(type_map == 1) | (type_map == 2) | (type_map == 9) | (type_map == 10)] = 1
        type_map[(type_map == 4) | (type_map == 5) | (type_map == 6) | (type_map == 7)] = 2
        type_map[(type_map == 8) | (type_map == 3)] = 3
    return type_map