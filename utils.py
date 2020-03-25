"""
Helper functions.
"""
import glob
import os
import SimpleITK as sitk
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from absl import flags
from typing import Callable


def save_plt_as_img(path: str, name: str, mri_slice: np.ndarray = None, seg: np.ndarray = None):
    """
    Merges a MRI slice and a segmentation map and saves it as a .jpg.
    :param path: Path where the out file should be saved
    :param name: Name of the out file
    :param mri_slice: 2D MRI-slice
    :param seg: 2D segmentation map
    """
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if mri_slice is not None:
        ax.imshow(mri_slice, 'gray', interpolation='none')
    if seg is not None:
        ax.imshow(seg, 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)
    plt.savefig(os.path.join(path, '{}.jpg'.format(name)), bbox_inches='tight')
    plt.close()


def collective_dice_smaller(best_eval_result: dict, current_eval_result: dict) -> bool:
    """
    Compares two evaluation results and returns true if the 2nd one is smaller.
    :param best_eval_result: best eval metrics.
    :param current_eval_result: current eval metrics.
    :return: True if the loss of current_eval_result is smaller; otherwise, False.
    """
    default_key = 'eval/classify_collective_dice'
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] > current_eval_result[default_key]


def decay_weights(cost: float, weight_decay_rate: float) -> float:
    """
    Calculates the loss for l2 weight decay and adds it to `cost`.
    :param cost: The loss before weight decay
    :param weight_decay_rate: Hyperparameter that regulates the effect of the weight decay
    :return: The loss after weight decay
    """
    costs = []
    for var in tf.compat.v1.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
    cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
    return cost


def get_estimator(FLAGS: flags.FlagValues, model_fn: Callable, epoch_steps: int) -> tf.estimator.Estimator:
    """
    Creates estimator.
    :param FLAGS: Input flags
    :param model_fn: Model function
    :param epoch_steps: Number of training steps per epoch
    :return: A Tensorflow estimator instance
    """
    # Estimator Configuration
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        keep_checkpoint_max=FLAGS.max_save,
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.save_steps
    )

    # Estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"model_dir": FLAGS.model_dir,
                'epoch_steps': epoch_steps})
    return estimator


def _dice_score(true: np.ndarray, pred: np.ndarray, brats_class: str) -> float:
    """
    Converts to a binary problem and calculates the Dice score from a
    predicted segmentation map given ground truth.
    :param true: Non-binary ground truth segmentation map
    :param pred: Non-binary predicted segmentation map
    :param brats_class: Binary problem
    :return: A Dice score
    """
    brats_class_mapping = {
        'whole': np.array([0, 1, 1, 1, 1]),
        'core': np.array([0, 1, 0, 1, 1]),
        'enhancing': np.array([0, 0, 0, 0, 1])
    }

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


def _get_ground_truth(patient_id: str, input_data_path: str) -> np.ndarray:
    """
    Get the ground truth segmentation map for a specific patient.
    :param patient_id: Patient ID
    :param input_data_path: Path to input data
    :return: Ground truth segmentation map
    """
    ground_truth = None

    for file in sorted(glob.glob(os.path.join(input_data_path, '**/{}_*'.format(patient_id)), recursive=True)):
        if 'seg' in file:
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(file))

    return ground_truth


def calc_and_export_standard_dice(pred_data_path: str):
    """
    Calculates and saves the standard Dice score.
    :param pred_data_path: Path to predicted segmentation maps
    """
    input_data_path = 'data/raw_data/downloaded_data/MICCAI_BraTS_2019_Data_Training'

    df = pd.DataFrame(columns=['Patient Id', 'Dice Whole', 'Dice Core', 'Dice Enhancing'])
    file_list = glob.glob(os.path.join(pred_data_path, '*.nii.gz'))

    for file in file_list:
        patient_id = file.split('/')[-1].split('.')[0]
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(file, sitk.sitkInt32))
        ground_truth = _get_ground_truth(patient_id, input_data_path)

        dice_whole = _dice_score(ground_truth, prediction, 'whole')
        dice_core = _dice_score(ground_truth, prediction, 'core')
        dice_enhancing = _dice_score(ground_truth, prediction, 'enhancing')

        df = df.append({'Patient Id': patient_id, 'Dice Whole': dice_whole, 'Dice Core': dice_core,
                        'Dice Enhancing': dice_enhancing}, ignore_index=True)

    out_path = os.path.join(pred_data_path, '..')
    df.to_csv(os.path.join(out_path, 'results.csv'), index=False, float_format='%.6f')
    df.describe().to_csv(os.path.join(out_path, 'results_summary.csv'), float_format='%.6f')

    print(df.describe())


def calc_and_export_equivariance_dice(pred_aug_data_path: str, aug_pred_data_path: str):
    """
    Calculates and saves the equivariance Dice score. See thesis report for more details.
    :param pred_aug_data_path: Path to augmented predictions
    :param aug_pred_data_path: Path to predicted augmentations
    """
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

        dice_whole = _dice_score(aug_pred, pred_aug, 'whole')
        dice_core = _dice_score(aug_pred, pred_aug, 'core')
        dice_enhancing = _dice_score(aug_pred, pred_aug, 'enhancing')

        df = df.append(
            {'Patient Id': pred_aug_patient_id, 'Equivariance Dice Whole': dice_whole,
             'Equivariance Dice Core': dice_core,
             'Equivariance Dice Enhancing': dice_enhancing}, ignore_index=True)

    out_path = os.path.join(pred_aug_data_path, '..')
    df.to_csv(os.path.join(out_path, 'equivariance_results.csv'), index=False, float_format='%.6f')
    df.describe().to_csv(os.path.join(out_path, 'equivariance_results_summary.csv'), float_format='%.6f')

    print(df.describe())
