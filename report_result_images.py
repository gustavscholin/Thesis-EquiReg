"""
Script for saving and showing result images for the master thesis report.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import SimpleITK as sitk
from utils import save_plt_as_img


def _save_patient_gt(patient_id: str, mri: np.ndarray, seg_map: np.ndarray):
    """
    Save ground truth images for a whole patient. Primarily used to make a .gif animation.
    :param patient_id: Id for the patient
    :param mri: Whole MRI-scan
    :param seg_map: Whole ground truth segmentation map
    """
    save_path = 'prediction_gifs/{}/ground_truth'.format(patient_id)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(mri.shape[0]):
        save_plt_as_img(save_path, str(i).zfill(3), mri[i, ...], seg_map[i, ...])
        print('{} : ground_truth : {}'. format(patient_id, i))


def _save_patient_pred(patient_id: str, model_name: str, mri: np.ndarray, prediction: np.ndarray):
    """
    Save prediction images for a whole patient. Primarily used to make a .gif animation.
    :param patient_id: Id for the patient
    :param model_name: Name of the predicting model
    :param mri: Whole MRI-scan
    :param prediction: Whole predicted segmentation map
    """
    save_path = 'prediction_gifs/{}/{}'.format(patient_id, model_name)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(mri.shape[0]):
        save_plt_as_img(save_path, str(i).zfill(3), mri[i, ...], prediction[i, ...])
        print('{} : {} : {}'.format(patient_id, model_name, i))


best_models_dict = {
    'best_0,01_baseline': 'baseline_0.01_2_seed_44',
    'best_0,01_equireg': 'equireg_0.01_1_seed_44',
    'best_0,05_baseline': 'baseline_0.05_2_seed_43',
    'best_0,05_equireg': 'equireg_0.05_1_seed_43',
    'best_0,1_baseline': 'baseline_0.1_1_seed_44',
    'best_0,1_equireg': 'equireg_0.1_1_seed_44',
    'best_1,0_baseline': 'baseline_1.0_3',
    'best_1,0_equireg': 'equireg_1.0_1'
}

if not os.path.isdir('report_images'):
    os.makedirs('report_images')

patient_ids = ['BraTS19_TMC_11964_1', 'BraTS19_2013_11_1', 'BraTS19_CBICA_BAN_1']
patient_slices = [80, 90, 75]

for i in range(3):
    data_path = glob.glob('data/raw_data/downloaded_data/MICCAI_BraTS_2019_Data_Training/**/{}'.format(patient_ids[i]))[
        0]

    mri = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, '{}_t1ce.nii.gz'.format(patient_ids[i]))))
    seg_map = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, '{}_seg.nii.gz'.format(patient_ids[i]))))
    seg_map = np.ma.masked_where(seg_map == 0, seg_map)
    # Save all ground truth patient slices for .gif animation
    _save_patient_gt(patient_ids[i], mri, seg_map)

    mri_img = mri[patient_slices[i], ...]
    seg_map_img = seg_map[patient_slices[i], ...]
    seg_map_img = np.ma.masked_where(seg_map_img == 0, seg_map_img)

    # Save and show specific ground truth slices of patients
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 5, 1)
    plt.title('Ground Truth')
    plt.imshow(mri_img, 'gray', interpolation='none')
    plt.imshow(seg_map_img, 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)
    save_plt_as_img('report_images', 'ground_truth_{}'.format(i), mri_slice=mri_img, seg=seg_map_img)

    # Save and show all models predicted version of the same specific patient slices
    for idx, model in enumerate(best_models_dict):
        model_path = glob.glob(os.path.join('data/predictions/results/**', best_models_dict[model]))[0]
        prediction_path = glob.glob(os.path.join(model_path, 'best*', 'standard', '{}.nii.gz'.format(patient_ids[i])))[
            0]

        pred_seg_map = sitk.GetArrayFromImage(sitk.ReadImage(prediction_path))
        pred_seg_map = np.ma.masked_where(pred_seg_map == 0, pred_seg_map)
        # Save all predicted patient slices for .gif animation
        _save_patient_pred(patient_ids[i], model, mri, pred_seg_map)

        pred_seg_map = pred_seg_map[patient_slices[i], ...]
        pred_seg_map = np.ma.masked_where(pred_seg_map == 0, pred_seg_map)
        save_plt_as_img('report_images', '{}_{}'.format(model, i), mri_slice=mri_img, seg=pred_seg_map)

        plt.subplot(2, 5, idx + 2)
        plt.title(model)
        plt.imshow(mri_img, 'gray', interpolation='none')
        plt.imshow(pred_seg_map, 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)

    plt.show(block=True)
