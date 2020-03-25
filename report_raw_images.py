"""
Script to show and save examples of raw MRI data for the master thesis report.
"""
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from utils import save_plt_as_img


if not os.path.isdir('report_images'):
    os.makedirs('report_images')

root_path = 'data/raw_data/downloaded_data/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_CBICA_BAN_1'
patient_slice = 90

data_paths = []
for root, dirnames, filenames in os.walk(root_path):
    if any('.nii.gz' in s for s in filenames):
        data_paths.append(root)

data_paths = sorted(data_paths)

for path in data_paths:
    print(path.split('/')[-1])
    mri = []
    seg_map = None

    for file in sorted(os.listdir(path)):
        if 'seg' in file:
            seg_map_sitk = sitk.ReadImage(os.path.join(path, file))
            seg_map = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file)))
        else:
            mri.append((sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file))), file.split('_')[-1].split('.')[0]))

    seg_map = np.ma.masked_where(seg_map == 0, seg_map)

    # Save and show an example of a MRI slice before histogram equalization
    data = mri[2][0].astype(np.float32)
    mask = (data != 0)
    plt.figure()
    plt.imshow(data[patient_slice, ...], 'gray', interpolation='none')
    save_plt_as_img('report_images', 'before_img', mri_slice=data[patient_slice, ...])
    data_flat = data[mask]
    plt.figure()
    plt.hist(data_flat)
    plt.savefig(os.path.join('report_images', 'before_hist.jpg'), bbox_inches='tight')
    n_bins = int(np.max(data_flat) + 1)

    # Histogram equalization
    histogram, bins = np.histogram(data_flat, n_bins, density=True)
    cdf = histogram.cumsum()
    cdf = cdf / cdf[-1]
    new_data_flat = np.interp(data_flat, bins[:-1], cdf)

    # Save and show an example of a MRI slice after histogram equalization
    plt.figure()
    plt.hist(new_data_flat)
    plt.savefig(os.path.join('report_images', 'after_hist.jpg'), bbox_inches='tight')
    data[mask] = new_data_flat
    plt.figure()
    plt.imshow(data[patient_slice, ...], 'gray', interpolation='none')
    save_plt_as_img('report_images', 'after_img', mri_slice=data[patient_slice, ...])

    # Save and show examples of all modalities of a single MRI-slice, one of those also exemplifying a MRI-slice
    # before preprocessing
    for seq in mri:
        save_plt_as_img('report_images', seq[1], mri_slice=seq[0][patient_slice, ...], seg=seg_map[patient_slice, ...])
        if seq[1] == 't1ce':
            save_plt_as_img('report_images', 'before_preprocess', mri_slice=seq[0][patient_slice, ...])

    for i in range(patient_slice, 155):
        plt.figure(num=str(i), figsize=(20, 10))
        for j in range(4):
            plt.subplot(1, 4, j+1)
            plt.title(mri[j][1])
            plt.imshow(mri[j][0][i, ...], 'gray', interpolation='none')
            plt.imshow(seg_map[i, ...], 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)
        plt.show(block=True)
