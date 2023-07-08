import os
import numpy as np
import scipy.io as sio


def size_calculate(major_id, inst_map):
    inst_size = np.sum((inst_map == (major_id + 1))>0)
    return inst_size


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def pick_minor_index(pool_minor, size_1, dataset):
    for basename, minor_class_list in pool_minor.items():
        ann = sio.loadmat(f'/Dataset/{dataset}/Train/Labels/{basename}.mat')
        inst_map = ann['inst_map']
        for minor_class_id in minor_class_list:
            mask = (inst_map == (minor_class_id+1)).astype(np.uint8) # +1을 해야 index가 맞음
            size_2 = np.sum(mask>0)
            if size_1 >= 2.3 * size_2:
                pool_minor[basename].remove(minor_class_id)
                return basename, ann, minor_class_id, pool_minor