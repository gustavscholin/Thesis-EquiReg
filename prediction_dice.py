import numpy as np
import os
import glob
import SimpleITK as sitk
import pandas as pd

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

    for file in os.listdir(pred_data_path):
        if '.nii.gz' not in file:
            continue
        patient_id = file.split('.')[0]
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_data_path, file), sitk.sitkInt32))
        ground_truth = get_input(patient_id, input_data_path)

        dice_whole = dice_score(ground_truth, prediction, 'whole')
        dice_core = dice_score(ground_truth, prediction, 'core')
        dice_enhancing = dice_score(ground_truth, prediction, 'enhancing')

        df = df.append({'Patient_ID': patient_id, 'Dice Whole': dice_whole, 'Dice Core': dice_core,
                        'Dice Enhancing': dice_enhancing}, ignore_index=True)

    df.to_csv(os.path.join(pred_data_path, 'results.csv'), index=False, float_format='%.6f')
    df.describe().to_csv(os.path.join(pred_data_path, 'results_summary.csv'), float_format='%.6f')

    print(df.describe())


def calc_and_export_consistency_dice(pred_data_path, aug_data_path):
    df = pd.DataFrame(
        columns=['Patient Id', 'Consistency Dice Whole', 'Consistency Dice Core', 'Consistency Dice Enhancing'])

    for file, aug_file in zip(os.listdir(pred_data_path), os.listdir(aug_data_path)):
        if '.nii.gz' not in file:
            continue
        patient_id = file.split('.')[0]
        aug_patient_id = aug_file.split('.')[0]
        assert (patient_id == aug_patient_id)

        prediction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_data_path, file), sitk.sitkInt32))
        aug_prediction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(aug_data_path, aug_file), sitk.sitkInt32))

        dice_whole = dice_score(aug_prediction, prediction, 'whole')
        dice_core = dice_score(aug_prediction, prediction, 'core')
        dice_enhancing = dice_score(aug_prediction, prediction, 'enhancing')

        df = df.append(
            {'Patient_ID': patient_id, 'Consistency Dice Whole': dice_whole, 'Consistency Dice Core': dice_core,
             'Consistency Dice Enhancing': dice_enhancing}, ignore_index=True)

    df.to_csv(os.path.join(pred_data_path, 'consistency_results.csv'), index=False, float_format='%.6f')
    df.describe().to_csv(os.path.join(pred_data_path, 'consistency_results_summary.csv'), float_format='%.6f')

    print(df.describe())

# experiment_name = '100_supervised_1'
# pred_data_path = os.path.join('data/predictions', 'baseline_1.0_supervised_2_best_val')
# calc_and_export_standard_dice(pred_data_path)
