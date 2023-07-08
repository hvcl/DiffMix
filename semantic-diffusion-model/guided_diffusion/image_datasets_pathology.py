import os
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    dataset_mode, # glysac
    data_dir, # /Dataset/glysac
    batch_size, # 4
    image_size, # 256
    class_cond=False, # True
    deterministic=False, # 
    random_crop=True,
    random_flip=True,
    is_train=True,
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
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'Train', 'Images' if is_train else 'Images'))
        mask_paths = _list_image_files_recursively(os.path.join(data_dir, 'Train', 'Labels' if is_train else 'Labels'))
    else:
        raise NotImplementedError('{} not implemented'.format(dataset_mode))

    print("Len of Dataset:", len(all_files))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        all_files,
        mask_paths,
        # classes=classes,
        # instances=instances,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        # random_crop=random_crop,
        # random_flip=random_flip,
        is_train=is_train
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


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
        resolution,
        image_paths,
        mask_paths,
        # classes=None,
        # instances=None,
        shard=0,
        num_shards=1,
        # random_crop=False,
        # random_flip=True,
        is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.images = image_paths[shard:][::num_shards]
        self.masks = mask_paths[shard:][::num_shards]
        # self.classes = classes
        # self.instances = instances
        # self.random_crop = random_crop
        # self.random_flip = random_flip

        if self.is_train:
            self.naug = 6 # orig*1 + rotate*3 + flip*2
        else:
            self.naug = 1

    def __len__(self):
        return len(self.images)*self.naug

    def __getitem__(self, idx):
        iaug = int(np.mod(idx, self.naug)) # 0~5
        index = int(np.floor(idx/self.naug)) # (idx*6-1) // 6
        
        path_img = self.images[index]
        path_mask = self.masks[index]
        
        with bf.BlobFile(path_img, "rb") as f:
            img = Image.open(f)
            img.load()
        img = np.array(img.convert("RGB"))
        
        with bf.BlobFile(path_mask, "rb") as f:
            if ".mat" in path_mask:
                ann = sio.loadmat(path_mask)
                ann_inst = np.expand_dims(ann["inst_map"], axis=0).astype(np.int32)
                ann_type = np.expand_dims(ann["type_map"], axis=0).astype(np.uint8)
                ann_type_orig = np.expand_dims(ann["type_map"], axis=0).astype(np.uint8)

                if self.dataset_mode == "consep":
                    ann_type[(ann_type == 3) | (ann_type == 4)] = 3
                    ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4
                elif self.dataset_mode == "glysac":
                    ann_type[(ann_type == 1) | (ann_type == 2) | (ann_type == 9) | (ann_type == 10)] = 1
                    ann_type[(ann_type == 4) | (ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 2
                    ann_type[(ann_type == 8) | (ann_type == 3)] = 3

        if self.is_train: # Data Aug
            np.random.seed()
            h, w, mod = img.shape
            img = img.astype(np.float32)
            img = img.transpose([2, 0, 1])
            
            # crop
            sh = np.random.randint(0, h-self.resolution-1)
            sw = np.random.randint(0, w-self.resolution-1)
            img = img[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]
            ann_type = ann_type[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]
            ann_inst = ann_inst[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]
            ann_type_orig = ann_type_orig[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]

            # aug (flip, rotate)
            if iaug<=3 and iaug>0: # iaug=1,2,3
                img = np.rot90(img, iaug, axes=(len(img.shape)-2, len(img.shape)-1))
                ann_inst = np.rot90(ann_inst, iaug, axes=(len(ann_inst.shape)-2, len(ann_inst.shape)-1))
                ann_type = np.rot90(ann_type, iaug, axes=(len(ann_type.shape)-2, len(ann_type.shape)-1))
                ann_type_orig = np.rot90(ann_type_orig, iaug, axes=(len(ann_type_orig.shape)-2, len(ann_type_orig.shape)-1))
            elif iaug>=4 and iaug<6: # iaug=4,5
                img = np.flip(img, len(img.shape)-(iaug-3))
                ann_inst = np.flip(ann_inst, len(ann_inst.shape)-(iaug-3))
                ann_type = np.flip(ann_type, len(ann_type.shape)-(iaug-3))
                ann_type_orig = np.flip(ann_type_orig, len(ann_type_orig.shape)-(iaug-3))
        else:
            np.random.seed()
            h, w, mod = img.shape
            img = img.astype(np.float32)
            img = img.transpose([2, 0, 1])
            # crop
            sh = np.random.randint(0, h-self.resolution-1)
            sw = np.random.randint(0, w-self.resolution-1)
            img = img[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]
            ann_type = ann_type[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]
            ann_inst = ann_inst[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]
            ann_type_orig = ann_type_orig[:, sh:(sh+self.resolution), sw:(sw+self.resolution)]

        img = img.astype(np.float32) / 127.5 - 1
        img = np.clip(img, -1, 1)
        img = img.copy()

        out_dict = {}
        out_dict["path"] = path_img
        out_dict["label"] = ann_type.copy()
        out_dict["instance"] = ann_inst.copy()
        out_dict["label_ori"] = ann_type_orig.copy() 

        return img, out_dict