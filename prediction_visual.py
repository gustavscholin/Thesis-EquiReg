"""
Script to visualize model predictions, both standard and equivariance.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import SimpleITK as sitk

from augmenters import img_aug


def get_input(patient_id: str, input_data_path: str) -> tuple:
    """
    Get the raw MRI and ground truth segmentation map (if it exists) of a specific patient.
    :param patient_id: The ID of the patient
    :param input_data_path: Path to the raw MRI data
    :return: MRI and ground truth segmentation map
    """
    mri = None
    ground_truth = None

    for file in sorted(glob.glob(os.path.join(input_data_path, '**/{}_*'.format(patient_id)), recursive=True)):
        if 'seg' in file:
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(file))
            ground_truth = np.ma.masked_where(ground_truth == 0, ground_truth)
        elif 't1ce' in file:
            mri = sitk.GetArrayFromImage(sitk.ReadImage(file))

    return mri, ground_truth


def standard_visual(pred_data_path: str, input_data_path: str):
    """
    Visualize the models standard prediction together with the ground truth (if it exists).
    The segmentation maps are shown slice by slice together with the raw input MRI slice in the background.
    :param pred_data_path: Path to the model prediction
    :param input_data_path: Path to the raw MRI data
    """
    nb_rows = 1 if pred_dataset == 'test' else 2

    prediction = sitk.GetArrayFromImage(sitk.ReadImage(pred_data_path))
    input_images, ground_truth = get_input(patient_id, input_data_path)
    prediction_masked = np.ma.masked_where(prediction == 0, prediction)

    for i in range(prediction.shape[0]):
        fig = plt.figure(figsize=(15, 7.5))
        fig.suptitle('{}: Slice {}'.format(patient_id, i))

        plt.subplot(nb_rows, 1, 1)
        plt.title('Prediction')
        plt.imshow(input_images[i, ...], 'gray', interpolation=None)
        plt.imshow(prediction_masked[i, ...], 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.4)

        if nb_rows == 2:
            plt.subplot(nb_rows, 1, 2)
            plt.title('Ground Truth')
            plt.imshow(input_images[i, ...], 'gray', interpolation=None)
            plt.imshow(ground_truth[i, ...], 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.4)

        plt.show(block=True)


def equivariance_visual(pred_aug_data_path: str, aug_pred_data_path: str, input_data_path: str):
    """
    Visualize the models equivariance predictions, that is, one augmented prediction and one predicted augmentation.
    The segmentation maps are shown slice by slice together with the augmented raw input MRI slice in the background.
    :param pred_aug_data_path: Path to the models augmented prediction
    :param aug_pred_data_path: Path to the models predicted augmentation
    :param input_data_path: Path to the raw MRI data
    """
    with open('data/processed_data/aug_seed.json', 'r') as fp:
        aug_seeds = json.load(fp)
    with open('data/processed_data/data_info.json', 'r') as fp:
        data_info = json.load(fp)

    input, _ = get_input(patient_id, input_data_path)

    # Augment the input images to match the predictions
    patient_index = [idx for idx, p in enumerate(data_info[pred_dataset]['paths']) if patient_id in p][0]
    patient_crop_idx = data_info[pred_dataset]['crop_idx'][patient_index]
    input_aug = input[patient_crop_idx[0]:patient_crop_idx[1], 8:232, 8:232]
    input_aug = img_aug(input_aug, aug_seeds[pred_dataset][patient_id], is_seg_maps=False)
    input_aug = np.pad(input_aug, ((0, 0), (8, 8), (8, 8)))
    input[patient_crop_idx[0]:patient_crop_idx[1]] = input_aug

    pred_aug = sitk.GetArrayFromImage(sitk.ReadImage(pred_aug_data_path, sitk.sitkInt32))
    aug_pred = sitk.GetArrayFromImage(sitk.ReadImage(aug_pred_data_path, sitk.sitkInt32))
    pred_aug_masked = np.ma.masked_where(pred_aug == 0, pred_aug)
    aug_pred_masked = np.ma.masked_where(aug_pred == 0, aug_pred)

    for i in range(input.shape[0]):
        fig = plt.figure(figsize=(15, 7.5))
        fig.suptitle('{}: Slice {}'.format(patient_id, i))

        plt.subplot(2, 1, 1)
        plt.title('Predicted Augmentation')
        plt.imshow(input[i, ...], 'gray', interpolation=None)
        plt.imshow(pred_aug_masked[i, ...], 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.4)

        plt.subplot(2, 1, 2)
        plt.title('Augmented Prediction')
        plt.imshow(input[i, ...], 'gray', interpolation=None)
        plt.imshow(aug_pred_masked[i, ...], 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.4)

        plt.show(block=True)


experiment_name = 'equireg_1.0_1'
pred_dataset = 'val'
patient_id = 'BraTS19_TCIA03_133_1'
mode = 'equivariance'
nb_channels = 1

if pred_dataset == 'test':
    input_data_path = 'data/raw_data/downloaded_data/MICCAI_BraTS_2019_Data_Validation'
else:
    input_data_path = 'data/raw_data/downloaded_data/MICCAI_BraTS_2019_Data_Training'

if mode == 'standard':
    pred_data_path = os.path.join('data/predictions/results', experiment_name.split('_')[0], experiment_name,
                                  'best_{}_prediction'.format(pred_dataset), 'standard', '{}.nii.gz'.format(patient_id))
    standard_visual(pred_data_path, input_data_path)
elif mode == 'equivariance':
    pred_aug_data_path = os.path.join('data/predictions/results', experiment_name.split('_')[0], experiment_name,
                                      'best_{}_prediction'.format(pred_dataset), 'pred_aug',
                                      '{}.nii.gz'.format(patient_id))
    aug_pred_data_path = os.path.join('data/predictions/results', experiment_name.split('_')[0], experiment_name,
                                      'best_{}_prediction'.format(pred_dataset), 'aug_pred',
                                      '{}.nii.gz'.format(patient_id))
    equivariance_visual(pred_aug_data_path, aug_pred_data_path, input_data_path)
