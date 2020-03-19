"""
Script to show and save examples of preprocessed MRI data for the master thesis report.
"""
import json
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import get_input_fn
from augmenters import img_aug, sup_aug
from utils import save_plt_as_img


patient_id = 'BraTS19_CBICA_BAN_1'
patient_slice = 90
strong_seed_seqs = ['298290031887607892964361995158779180246', '295631042280730438349207938294089137910',
                    '210613237776179382952667370229095541249']
weak_seed_seqs = ['283915260950354392800265287473656160289', '288272994038709529613915206346046964958',
                  '74636310939612720933168116878377399537']
data_dir = 'data/processed_data'

if not os.path.isdir('report_images'):
    os.makedirs('report_images')

with open(os.path.join(data_dir, 'data_info.json'), 'r') as fp:
    data_info = json.load(fp)

input_fcn = get_input_fn(data_dir, 'val', data_info, batch_size=1, sup_cut=1.0, unsup_cut=0.0, unsup_ratio=0)
dataset = input_fcn()

patient_index = [idx for idx, p in enumerate(data_info['val']['paths']) if patient_id in p][0]
img_index = np.cumsum(data_info['val']['slices'])[patient_index - 1] + patient_slice

for idx, sample in enumerate(dataset):

    if idx >= img_index:

        # Save and show an example of a preprocessed MRI slice
        img_np = sample['image'].numpy()[0, ...]
        seg_mask_np = tf.cast(sample['seg_mask'], tf.uint8).numpy()[0, ...]
        seg_mask_np = np.ma.masked_where(seg_mask_np == 0, seg_mask_np)

        save_plt_as_img('report_images', 'after_preprocess', img=img_np[..., 2])

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np[..., 2], 'gray', interpolation='none')
        plt.subplot(1, 2, 2)
        plt.imshow(img_np[..., 2], 'gray', interpolation='none')
        plt.imshow(seg_mask_np, 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)

        # Save one original
        save_plt_as_img('report_images', 'original', img=img_np[..., 2],
                        seg=seg_mask_np)

        # Save and show 3 examples of strong augmentations of the original
        strong_aug_imgs = img_aug(np.stack([img_np[..., 2]] * 3), strong_seed_seqs, is_seg_maps=False)
        strong_aug_seg_masks = img_aug(np.stack([seg_mask_np] * 3), strong_seed_seqs, is_seg_maps=True)
        strong_aug_seg_masks = np.ma.masked_where(strong_aug_seg_masks == 0, strong_aug_seg_masks)

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 4, 1)
        plt.imshow(img_np[..., 2], 'gray', interpolation='none')
        plt.imshow(seg_mask_np, 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)
        for i in range(3):
            save_plt_as_img('report_images', 'strong_aug_example_{}'.format(str(i)), img=strong_aug_imgs[i, ...],
                            seg=strong_aug_seg_masks[i, ...])
            plt.subplot(1, 4, i + 2)
            plt.imshow(strong_aug_imgs[i, ...], 'gray', interpolation='none')
            plt.imshow(strong_aug_seg_masks[i, ...], 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)

        # Save and show 3 examples of weak augmentations of the original
        weak_aug_imgs = []
        weak_aug_seg_masks = []
        for i in range(3):
            weak_aug_img, weak_aug_seg_mask = sup_aug(img_np[..., 2], seg_mask_np, entropy=int(weak_seed_seqs[i]))
            weak_aug_imgs.append(weak_aug_img)
            weak_aug_seg_masks.append(np.ma.masked_where(weak_aug_seg_mask == 0, weak_aug_seg_mask))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 4, 1)
        plt.imshow(img_np[..., 2], 'gray', interpolation='none')
        plt.imshow(seg_mask_np, 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)
        for i in range(3):
            save_plt_as_img('report_images', 'weak_aug_example_{}'.format(str(i)), img=weak_aug_imgs[i],
                            seg=weak_aug_seg_masks[i])
            plt.subplot(1, 4, i + 2)
            plt.imshow(weak_aug_imgs[i], 'gray', interpolation='none')
            plt.imshow(weak_aug_seg_masks[i], 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)

        plt.show(block=True)
