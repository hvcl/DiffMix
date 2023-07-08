import os
import cv2
import glob
import tqdm
import random
import numpy as np
import skimage.filters
import scipy.io as sio

from math import sqrt as sqrt
from skimage.morphology import disk
from scipy.ndimage import binary_dilation, distance_transform_cdt

from gradmix_scripts.utils import size_calculate, bounding_box, pick_minor_index

# Set target dataset
dataset = 'glysac'

if dataset=='glysac':
    img_pths = glob.glob(f'/Dataset/{dataset}/Train/Images/*.tif')
elif dataset=='consep':
    img_pths = glob.glob(f'/Dataset/{dataset}/Train/Images/*.png')
seg_pths = glob.glob(f'/Dataset/{dataset}/Train/Labels/*.mat')

save_img_path = f'./data/{dataset}/Grad_mix_Images'
save_inp_path = f'./data/{dataset}/Grad_mix_Inpainted'
save_seg_path = f'./data/{dataset}/Grad_mix_Labels'
os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_inp_path, exist_ok=True)
os.makedirs(save_seg_path, exist_ok=True)

for (img_pth, seg_pth) in tqdm.tqdm(zip(img_pths, seg_pths)):
    print(img_pth, seg_pth)
    
    if dataset=='glysac':
        img_list = glob.glob(f'/Dataset/{dataset}/Train/Images/*.tif')
    elif dataset=='consep':
        img_list = glob.glob(f'/Dataset/{dataset}/Train/Images/*.png')
    random.shuffle(img_list)
    img_list.remove(img_pth)

    filename = os.path.basename(img_pth)
    basename = filename.split('.')[0]

    img = cv2.imread(img_pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    frame = img.copy()
    final = img.copy()
    inpainted = img.copy()

    ann = sio.loadmat(seg_pth)
    inst_map = ann['inst_map']
    inst_map_out = inst_map.copy()
    type_map = ann['type_map']
    class_arr = np.squeeze(ann['inst_type'])

    # Combine nuclei classes: you can skip depends upon your data
    if dataset == 'glysac':
        class_arr[(class_arr == 1) | (class_arr == 2) | (class_arr == 9) | (class_arr == 10)] = 1
        class_arr[(class_arr == 4) | (class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 2
        class_arr[(class_arr == 8) | (class_arr == 3)] = 3
    elif dataset == 'consep':
        class_arr[(class_arr == 3) | (class_arr == 4)] = 3
        class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 4
    class_arr_copy = class_arr.copy()

    # x, y
    # cent to 0 if cell is located at the corner
    cent_ann = ann['inst_centroid'] # x, y
    for i, cent in enumerate(cent_ann):
        if ((cent[1] < 30) or
            (cent[1] > (inst_map.shape[0]-30)) or
            (cent[0] < 30) or
            (cent[0] > (inst_map.shape[1]-30))):
            class_arr_copy[i] = 0
    nuc_color = img * (inst_map[..., np.newaxis]>0)
    # avg nuclear color intensities
    avg_color_1 = [
        np.sum(nuc_color[...,0]) / np.sum(nuc_color[...,0]>0), 
        np.sum(nuc_color[...,1]) / np.sum(nuc_color[...,1]>0), 
        np.sum(nuc_color[...,2]) / np.sum(nuc_color[...,2]>0)
    ]

    # Calculate Major and Minor class indices
    major_class_idx = list(np.where(class_arr_copy == 2)[0]) + \
                        list(np.where(class_arr_copy == 3)[0]) + \
                        list(np.where(class_arr_copy == 4)[0])
    picked_major_class = list(np.random.choice(major_class_idx, int(0.8 * len(major_class_idx)), replace=False))
    picked_major_class = sorted(picked_major_class, key=lambda x: size_calculate(x, inst_map))

    minor_class_idx = list(np.where(class_arr_copy == 1)[0])
    pool_minor = {}
    pool_minor[basename] = minor_class_idx

    for img_pth, in zip(img_list):
        file_name = os.path.basename(img_pth)
        basename_1 = file_name.split('.')[0]
        ann_2 = sio.loadmat(f'/Dataset/{dataset}/Train/Labels/{basename_1}.mat')
        inst_map_2 = ann_2['inst_map']

        class_arr_2 = np.squeeze(ann_2['inst_type'])
        if dataset == 'glysac':
            class_arr_2[(class_arr_2 == 1) | (class_arr_2 == 2) | (class_arr_2 == 9) | (class_arr_2 == 10)] = 1
            class_arr_2[(class_arr_2 == 4) | (class_arr_2 == 5) | (class_arr_2 == 6) | (class_arr_2 == 7)] = 2
            class_arr_2[(class_arr_2 == 8) | (class_arr_2 == 3)] = 3
        elif dataset == 'consep':
            class_arr_2[(class_arr_2 == 3) | (class_arr_2 == 4)] = 3
            class_arr_2[(class_arr_2 == 5) | (class_arr_2 == 6) | (class_arr_2 == 7)] = 4
        cent_ann_2 = ann_2['inst_centroid']

        for i, cent in enumerate(cent_ann_2):
            if ((cent[1] < 30) or 
                    (cent[1] > (inst_map_2.shape[0]-30)) or 
                    (cent[0] < 30) or 
                    (cent[0] > (inst_map_2.shape[1]-30))):
                class_arr_2[i] = 0
        minor_class_idx = list(np.where(class_arr_2 == 1)[0])
        pool_minor[basename_1] = minor_class_idx
    
    for major_class_idx in picked_major_class:
        mask_0 = (inst_map == (major_class_idx+1)).astype(np.uint8)        
        mask = binary_dilation(mask_0, iterations=2).astype(np.uint8)
        cent1 = cent_ann[major_class_idx]
        bbox1 = bounding_box(mask) # [rmin, rmax, cmin, cmax]
        h1, w1 = bbox1[1] - bbox1[0], bbox1[3] - bbox1[2]
        size_1 = np.sum(mask>0)

        try:
            basename_2, ann_2, index_2, pool_minor = pick_minor_index(pool_minor, size_1, dataset)
        except TypeError:
            continue
        # if dataset=='glysac':
        #     img_2_ori = cv2.imread(f'/Dataset/{dataset}/Train/Images/{basename_2}.tif')
        # elif dataset=='consep':
        img_2_ori = cv2.imread(f'/Dataset/{dataset}/Train/Images/{basename_2}.png')

        img_2_ori = cv2.cvtColor(img_2_ori, cv2.COLOR_BGR2RGB)
        img_2 = img_2_ori.copy()
        inst_map_2 = ann_2['inst_map']
        cent_ann_2 = ann_2['inst_centroid']
        mask_2 = (inst_map_2 == (index_2+1)).astype(np.uint8)
        cent_2 = cent_ann_2[index_2]
        bbox2 = bounding_box(mask_2)
        h2, w2 = bbox2[1] - bbox2[0], bbox2[3] - bbox2[2]

        img_2_for_frame = img_2.copy()
        img_2[...,0][mask_2 > 0] = (img_2_ori[...,0][mask_2 > 0] + avg_color_1[0]) / 2
        img_2[...,1][mask_2 > 0] = (img_2_ori[...,1][mask_2 > 0] + avg_color_1[1]) / 2
        img_2[...,2][mask_2 > 0] = (img_2_ori[...,2][mask_2 > 0] + avg_color_1[2]) / 2

        class_arr[major_class_idx] = 1

        # Inapinting
        # inpaint(final, mask, eps)
        # inpaint(inpainted, mask, eps)

        inst_map_out[inst_map == (major_class_idx+1)] = 0
        img_copy = img.copy()
        img_copy[bbox1[0]:bbox1[1], bbox1[2]:bbox1[3], :] = \
            img_2[
                int(np.round(cent_2[1])-h1/2):int(np.round(cent_2[1])+h1/2),
                int(np.round(cent_2[0])-w1/2):int(np.round(cent_2[0])+w1/2), 
                :
            ]
        img_frame = img.copy()
        img_frame[bbox1[0]:bbox1[1], bbox1[2]:bbox1[3], :] = \
            img_2_for_frame[
                int(np.round(cent_2[1])-h1/2):int(np.round(cent_2[1])+h1/2),
                int(np.round(cent_2[0])-w1/2):int(np.round(cent_2[0])+w1/2), 
                :
            ]
        
        mask_translated = np.zeros_like(mask)
        mask_translated[
            int(np.round(cent1[1])-h2/2):int(np.round(cent1[1])+h2/2), 
            int(np.round(cent1[0])-w2/2):int(np.round(cent1[0])+w2/2)
            ] = mask_2[bbox2[0]:bbox2[1], bbox2[2]:bbox2[3]]

        inst_map_out[mask_translated > 0] = major_class_idx + 1
        mask = ((mask + mask_translated)>0).astype(np.uint8)
        mask_substract = mask - mask_translated
        cdt_map = distance_transform_cdt(1 - mask_translated).astype('float32')
        cdt_map[mask==0] = 0
        cdt_map[mask_substract>0] -= 1
        cdt_map[mask_substract>0] /= np.amax(cdt_map[mask_substract>0])
        cdt_map[mask_substract>0] = 1 - cdt_map[mask_substract>0]
        cdt_map[mask_translated > 0] = 1

        ######## GRADMIX는 여기부터 다시
        final = final*(1-mask[...,np.newaxis]) + \
                    img_copy*mask_translated[...,np.newaxis] + \
                    (img_copy*(cdt_map*mask_substract)[...,np.newaxis]).astype(np.uint8) + \
                    (final*((1-cdt_map)*mask_substract)[...,np.newaxis]).astype(np.uint8)
        final = (img_copy * cdt_map[...,np.newaxis]).astype(np.uint8) + \
                (final * (1 - cdt_map)[...,np.newaxis]).astype(np.uint8)
        final_smooth = np.stack([
            skimage.filters.median(final[...,0], disk(1)), 
            skimage.filters.median(final[...,1], disk(1)), 
            skimage.filters.median(final[...,2], disk(5))
            ], axis=2)
        final = (final * (1 - mask_substract[...,np.newaxis])).astype(np.uint8) + \
                (final_smooth.astype(np.float32) * mask_substract[...,np.newaxis]).astype(np.uint8)
    
        # frame

    final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    type_map = np.zeros_like(type_map)
    inst_list = list(np.unique(inst_map_out))
    inst_list.remove(0)
    inst_type = []
    for inst_id in inst_list:
        type_map[inst_map_out == int(inst_id)] = class_arr[int(inst_id) - 1]
        inst_type.append(class_arr[int(inst_id-1)])
    
    cv2.imwrite(f'{save_img_path}/{basename}_synthesized.png', final)
    cv2.imwrite(f'{save_inp_path}/{basename}_inpainted.png', inpainted)
    sio.savemat(f'{save_seg_path}/{basename}_synthesized.mat', {
                    'inst_map'      : inst_map_out,
                    'type_map'      : type_map,
                    'inst_type'     : np.array(class_arr[:, None]), 
                    'inst_centroid' : cent_ann,
                    })