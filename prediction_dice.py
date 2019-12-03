import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import SimpleITK as sitk

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

    score = (2 * tp) / (2 * tp + fp + fn)

    return score


def get_input(patient_id, input_data_path):
    images = []
    ground_truth = None

    for file in sorted(glob.glob(os.path.join(input_data_path, '**/{}_*'.format(patient_id)), recursive=True)):
        if 'seg' in file:
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(file))
        else:
            images.append(sitk.GetArrayFromImage(sitk.ReadImage(file)))
    images = np.stack(images)

    return images, ground_truth


def calc_and_export_dice(pred_data_path):
    input_data_path = 'data/raw_data/downloaded_data/MICCAI_BraTS_2019_Data_Training'

    dice_whole = {}
    dice_core = {}
    dice_enhancing = {}

    for file in os.listdir(pred_data_path):
        if '.nii.gz' not in file:
            continue
        patient_id = file.split('.')[0]
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_data_path, file), sitk.sitkInt32))
        input_images, ground_truth = get_input(patient_id, input_data_path)

        dice_whole[patient_id] = dice_score(ground_truth, prediction, 'whole')
        dice_core[patient_id] = dice_score(ground_truth, prediction, 'core')
        dice_enhancing[patient_id] = dice_score(ground_truth, prediction, 'enhancing')

    with open(os.path.join(pred_data_path, 'results.txt'), 'w') as txt_file:
        print('Whole Dice: {}'.format(np.mean(list(dice_whole.values()))), file=txt_file)
        print('Core Dice: {}'.format(np.mean(list(dice_core.values()))), file=txt_file)
        print('Enhancing Dice: {}'.format(np.mean(list(dice_enhancing.values()))), file=txt_file)

    print('Whole Dice: {}'.format(np.mean(list(dice_whole.values()))))
    print('Core Dice: {}'.format(np.mean(list(dice_core.values()))))
    print('Enhancing Dice: {}'.format(np.mean(list(dice_enhancing.values()))))


# experiment_name = '100_supervised_1'
# pred_data_path = os.path.join('data/predictions', experiment_name, 'latest_ckpt_val_prediction')
# calc_and_export_dice(pred_data_path)
