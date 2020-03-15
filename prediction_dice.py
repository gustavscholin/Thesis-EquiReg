import json
import numpy as np
import os
import glob
import SimpleITK as sitk
import pandas as pd
import augmenters

brats_class_mapping = {
    'whole': np.array([0, 1, 1, 1, 1]),
    'core': np.array([0, 1, 0, 1, 1]),
    'enhancing': np.array([0, 0, 0, 0, 1])
}


def dice_score(true, pred, brats_class):
    true_bool = brats_class_mapping[brats_class][true]
    pred_bool = brats_class_mapping[brats_class][pred]

    tp = np.sum(np.logical_and(true_bool == 1, pred_bool == 1))
    fp = np.sum(np.logical_and(true_bool == 0, pred_bool == 1))
    fn = np.sum(np.logical_and(true_bool == 1, pred_bool == 0))

    if tp + fp + fn == 0:
        score = 1.0
    else:
        score = (2 * tp) / (2 * tp + fp + fn)

    return score


def get_input(patient_id, input_data_path):
    ground_truth = None

    for file in sorted(glob.glob(os.path.join(input_data_path, '**/{}_*'.format(patient_id)), recursive=True)):
        if 'seg' in file:
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(file))

    return ground_truth


def calc_and_export_standard_dice(pred_data_path):
    input_data_path = 'data/raw_data/downloaded_data/MICCAI_BraTS_2019_Data_Training'

    df = pd.DataFrame(columns=['Patient Id', 'Dice Whole', 'Dice Core', 'Dice Enhancing'])
    file_list = glob.glob(os.path.join(pred_data_path, '*.nii.gz'))

    for file in file_list:
        patient_id = file.split('/')[-1].split('.')[0]
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(file, sitk.sitkInt32))
        ground_truth = get_input(patient_id, input_data_path)

        dice_whole = dice_score(ground_truth, prediction, 'whole')
        dice_core = dice_score(ground_truth, prediction, 'core')
        dice_enhancing = dice_score(ground_truth, prediction, 'enhancing')

        df = df.append({'Patient Id': patient_id, 'Dice Whole': dice_whole, 'Dice Core': dice_core,
                        'Dice Enhancing': dice_enhancing}, ignore_index=True)

    out_path = os.path.join(pred_data_path, '..')
    df.to_csv(os.path.join(out_path, 'results.csv'), index=False, float_format='%.6f')
    df.describe().to_csv(os.path.join(out_path, 'results_summary.csv'), float_format='%.6f')

    print(df.describe())


def calc_and_export_equivariance_dice(pred_aug_data_path, aug_pred_data_path):
    df = pd.DataFrame(
        columns=['Patient Id', 'Equivariance Dice Whole', 'Equivariance Dice Core', 'Equivariance Dice Enhancing'])

    pred_aug_file_list = glob.glob(os.path.join(pred_aug_data_path, '*.nii.gz'))
    aug_pred_file_list = glob.glob(os.path.join(aug_pred_data_path, '*.nii.gz'))

    for pred_aug_file, aug_pred_file in zip(pred_aug_file_list, aug_pred_file_list):
        pred_aug_patient_id = pred_aug_file.split('/')[-1].split('.')[0]
        aug_pred_patient_id = aug_pred_file.split('/')[-1].split('.')[0]
        assert (pred_aug_patient_id == aug_pred_patient_id)

        pred_aug = sitk.GetArrayFromImage(sitk.ReadImage(pred_aug_file, sitk.sitkInt32))
        aug_pred = sitk.GetArrayFromImage(sitk.ReadImage(aug_pred_file, sitk.sitkInt32))

        dice_whole = dice_score(aug_pred, pred_aug, 'whole')
        dice_core = dice_score(aug_pred, pred_aug, 'core')
        dice_enhancing = dice_score(aug_pred, pred_aug, 'enhancing')

        df = df.append(
            {'Patient Id': pred_aug_patient_id, 'Equivariance Dice Whole': dice_whole, 'Equivariance Dice Core': dice_core,
             'Equivariance Dice Enhancing': dice_enhancing}, ignore_index=True)

    out_path = os.path.join(pred_aug_data_path, '..')
    df.to_csv(os.path.join(out_path, 'equivariance_results.csv'), index=False, float_format='%.6f')
    df.describe().to_csv(os.path.join(out_path, 'equivariance_results_summary.csv'), float_format='%.6f')

    print(df.describe())

# experiment_name = '100_supervised_1'
# pred_data_path = os.path.join('data/predictions', '1576212472_val_prediction/standard')
# aug_data_path = os.path.join('data/predictions', '1576212472_val_prediction/aug')
# calc_and_export_consistency_dice(pred_data_path, aug_data_path, 'val')
