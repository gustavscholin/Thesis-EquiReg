"""
Script for preprocessing of the BraTS 2019 dataset.
"""
import json
import random
import sys
import numpy as np
import SimpleITK as sitk
import os
import tensorflow as tf
import itertools

from absl import flags, app, logging
from functools import partial
from multiprocessing import Pool

DOWNLOADED_DATA_PATH = 'data/raw_data/downloaded_data'
PROCESSED_DATA_PATH = 'data/processed_data'

LABELED_DATA_FOLDER = 'MICCAI_BraTS_2019_Data_Training'
UNLABELED_DATA_FOLDER = 'MICCAI_BraTS_2019_Data_Validation'

FLAGS = flags.FLAGS


def _float_feature(value: np.ndarray) -> tf.train.Feature:
    """
    Get a Tensorflow float feature from a float numpy array,
    for saving as .tfrecord.
    :param value: Float numpy array
    :return: Tensorflow float feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int_feature(value: np.ndarray) -> tf.train.Feature:
    """
    Get a Tensorflow int feature from a int numpy array,
    for saving as .tfrecord.
    :param value: Int numpy array
    :return: Tensorflow int feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def save_tfrecord(example_list: list, out_path: str, shard_cnt: int):
    """
    Saves list of Tensorflow examples as a .tfrecord-file.
    :param example_list: List of Tensorflow examples
    :param out_path: Path to save file to
    :param shard_cnt: Integer to number the file
    """
    record_writer = tf.io.TFRecordWriter("{}.{:d}".format(out_path, shard_cnt))
    for example in example_list:
        record_writer.write(example.SerializeToString())
    record_writer.close()
    logging.info("Saved {} examples to {}".format(len(example_list), out_path))


def save_npy(data_list: list, out_dir: str, shard_cnt: int):
    """
    Saves list of numpy arrays as a .npy-file.
    :param data_list: List of numpy arrays
    :param out_dir: Path to save file to
    :param shard_cnt: Integer to number the file
    """
    if data_list:
        data = np.concatenate(data_list, axis=0)
        out_path = '{}_{}.npy'.format(out_dir, shard_cnt)
        np.save(out_path, data)
        logging.info('Saved {} samples to {}'.format(data.shape[0], out_path))


def get_example_list(images: np.ndarray, seg_maps: np.ndarray) -> list:
    """
    Get list of Tensorflow examples from images and segmentation maps.
    :param images: Numpy array of images
    :param seg_maps: Numpy array of segmentation maps
    :return: List of Tensorflow examples
    """
    example_list = []
    if seg_maps is not None:
        for image, seg_map in zip(images, seg_maps):
            feature = {
                'image': _float_feature(image.reshape(-1)),
                'seg_map': _int_feature(seg_map.reshape(-1))
            }
            example_list.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    else:
        for image in images:
            feature = {
                'image': _float_feature(image.reshape(-1))
            }
            example_list.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return example_list


def preprocess_data(data_paths: list, out_path: str, split: str, out_format: str, nbr_cores: int = 1) -> tuple:
    """
    Preprocess the raw patient MRI-scans. The preprocessing is distributed on the number of CPU-cores specified.
    The function returns a tuple containing containing information for MRI-scans;
    path to raw data, number of 2D-images it included, the interval of included 2D-images and
    augmentation seed for each 2D-image.
    :param data_paths: List of paths to raw data samples (MRI-scans + segmentation maps).
    :param out_path: Path where to save the preprocessed data
    :param split: Whether it is training, validation or test data
    :param out_format: Whether to save as .tfrecords or .npy
    :param nbr_cores: The number of CPU-cores to distribute the work
    :return: A tuple with path to raw data, number of 2D-images it included,
    the interval of included 2D-images and augmentation seed for each 2D-image,
    for each MRI-scan
    """
    logging.info('Pre-processing {} data'.format(split))

    out_path = os.path.join(out_path, split)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    partial_preproc_one = partial(preproc_one, out_format=out_format, split=split)

    pool = Pool(processes=nbr_cores)
    shard_cnt = 0

    path_list = []
    nb_slices_list = []
    crop_idx_list = []
    aug_seeds_dict = {}

    if out_format == 'tfrecord':
        example_lists = []
        file_name = os.path.join(out_path, "{}_data.tfrecord".format(split))

        for sample_dict in pool.imap(partial_preproc_one, data_paths):
            example_lists.append(sample_dict['example_list'])
            path_list.append(sample_dict['path'])
            nb_slices_list.append(sample_dict['nb_slices'])
            crop_idx_list.append(sample_dict['crop_idx'])
            aug_seeds_dict[sample_dict['path'].split('/')[-1]] = sample_dict['aug_seeds']

            if len(example_lists) == FLAGS.nb_patients_per_file or len(data_paths) == len(path_list):
                examples = list(itertools.chain.from_iterable(example_lists))
                save_tfrecord(examples, file_name, shard_cnt)
                shard_cnt += 1
                example_lists = []

    elif out_format == 'numpy':
        data_list = []
        images_file_path = os.path.join(out_path, '{}_images'.format(split))
        seg_masks_file_path = os.path.join(out_path, '{}_seg_masks'.format(split))

        for sample_dict in pool.imap(partial_preproc_one, data_paths):
            data_list.append((sample_dict['images'], sample_dict['seg_masks']))
            path_list.append(sample_dict['path'])
            nb_slices_list.append(sample_dict['nb_slices'])
            crop_idx_list.append(sample_dict['crop_idx'])
            aug_seeds_dict[sample_dict['path'].split('/')[-1]] = sample_dict['aug_seeds']

            if len(data_list) == FLAGS.nb_patients_per_file or len(data_paths) == len(path_list):
                images, seg_masks = zip(*data_list)
                save_npy(images, images_file_path, shard_cnt)
                if seg_masks[0] is not None:
                    save_npy(seg_masks, seg_masks_file_path, shard_cnt)
                shard_cnt += 1
                data_list = []

    return path_list, nb_slices_list, crop_idx_list, aug_seeds_dict


def get_data_paths(path: str) -> list:
    """
    Get list of paths to all directories containing .nii.gz-files
    in path.
    :param path: Path where to look for files
    :return: List of paths
    """
    data_paths = []
    for root, dirnames, filenames in os.walk(path):
        if any('.nii.gz' in s for s in filenames):
            data_paths.append(root)
    return sorted(data_paths)


def preproc_one(path: str, out_format: str, split: str) -> dict:
    """
    Function to preprocess one MRI-scan. The preprocessing includes
    cropping, bias field correction and histogram equalization. A augmentation
    seed is also generated for each 2D-image in the scan to be used for the equivariance
    Dice score.
    :param path: Path to raw MRI-scan
    :param out_format: Whether to save as .tfrecords or .npy
    :param split: Whether it is training, validation or test data
    :return: A dictionary containing; Tensor flow examples list or 2D-images and
    segmentation maps as numpy arrays, number of 2D-images, path to the raw data,
    interval of 2D-images not cropped away, augmentation seeds.
    """
    logging.info('Processing {}'.format(path))
    mri_modalities = []
    seg_masks = None
    for file in sorted(os.listdir(path)):
        if 'seg' not in file:
            mri_modalities.append(sitk.ReadImage(os.path.join(path, file), sitk.sitkFloat32))
        else:
            seg_masks = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file), sitk.sitkInt64))
            seg_masks[np.where(seg_masks == 4)] = 3  # Moving all class 4 to class 3. Class 3 isn't used in BraTS 2019

    mri_modalities, seg_masks, crop_idx = crop(mri_modalities, seg_masks, split)

    for i in range(4):
        mri_modalities[i] = bias_field_correction(mri_modalities[i])

    images = sitk.GetArrayFromImage(sitk.Compose(mri_modalities))

    for i in range(4):
        images[:, :, :, i] = histogram_equalization(images[:, :, :, i])

    aug_seeds = []
    for i in range(images.shape[0]):
        sq = np.random.SeedSequence()
        aug_seeds.append(sq.entropy)

    if out_format == 'tfrecord':
        return {
            'example_list': get_example_list(images, seg_masks),
            'nb_slices': images.shape[0],
            'path': path,
            'crop_idx': crop_idx,
            'aug_seeds': aug_seeds
        }

    elif out_format == 'numpy':
        return {
            'images': images,
            'seg_masks': seg_masks,
            'nb_slices': images.shape[0],
            'path': path,
            'crop_idx': crop_idx,
            'aug_seeds': aug_seeds
        }


def crop(mri_modalities: list, seg_map: np.ndarray, split: str) -> tuple:
    """
    Crops MRI-scan to remove all-zero 2D-images and resize remaining 2D-images from
    240x240 to 224x224.
    :param mri_modalities: Each of the MRI-scan's modalities in a list
    :param seg_map: MRI-scan's ground truth segmentation map
    :param split: Whether it is training, validation or test data
    :return: Cropped versions of the modalities and the segmentation map
    """
    if not split == 'test':
        min_idx = 155
        max_idx = 0
        for mri_channel in mri_modalities:
            mask = mri_channel != 0
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(mask)
            bounding_box = label_shape_filter.GetBoundingBox(1)

            if bounding_box[2] < min_idx:
                min_idx = bounding_box[2]
            if bounding_box[5] + bounding_box[2] > max_idx:
                max_idx = bounding_box[5] + bounding_box[2]

        # Crop mri and seg-mask to 224x224 to fit Tiramisu model
        return [mri_channel[8:232, 8:232, min_idx:max_idx] for mri_channel in mri_modalities], \
               seg_map[min_idx:max_idx, 8:232, 8:232], (min_idx, max_idx)
    else:
        return [mri_channel[8:232, 8:232, :] for mri_channel in mri_modalities], None, (0, 155)


def bias_field_correction(mri: sitk.Image) -> sitk.Image:
    """
    Performs bias field correction on one MRI-scan modality. See SimpleITK documentation for details.
    :param mri: MRI-scan
    :return: Corrected MRI-scan
    """
    mask = (mri != 0)
    mri = sitk.Cast(mri, sitk.sitkFloat32)
    return sitk.N4BiasFieldCorrection(mri, mask)


def histogram_equalization(mri: np.ndarray) -> np.ndarray:
    """
    Performs histogram equalization on one MRI-scan modality.
    :param mri: MRI-scan
    :return: Equalized MRI-scan
    """
    mask = (mri != 0)
    data_flat = mri[mask]

    n_bins = int(np.max(data_flat) + 1)

    histogram, bins = np.histogram(data_flat, n_bins, density=True)

    cdf = histogram.cumsum()
    cdf = cdf / cdf[-1]

    mri[mask] = np.interp(data_flat, bins[:-1], cdf)

    return mri


def main(_):
    """
    Main method for the preprocessing.
    """
    labeled_data_root = os.path.join(FLAGS.in_path, LABELED_DATA_FOLDER)
    unlabeled_data_root = os.path.join(FLAGS.in_path, UNLABELED_DATA_FOLDER)

    if not os.path.exists(labeled_data_root):
        logging.fatal('Input data not found')
        sys.exit()

    if not os.path.exists(FLAGS.out_path) or not os.listdir(FLAGS.out_path):
        logging.info('Starting pre-processing')

        if not os.path.exists(FLAGS.out_path):
            os.makedirs(FLAGS.out_path)

        # Split up the labeled data in training and validation
        labeled_data_paths = get_data_paths(labeled_data_root)
        random.Random(FLAGS.seed).shuffle(labeled_data_paths)
        nb_train_paths = int(len(labeled_data_paths) * FLAGS.train_cut)
        train_paths = labeled_data_paths[:nb_train_paths]
        val_paths = labeled_data_paths[nb_train_paths:]

        test_paths = get_data_paths(unlabeled_data_root)

        train_paths, train_slices, train_crops, train_aug_seeds = preprocess_data(train_paths, FLAGS.out_path,
                                                                                  split="train",
                                                                                  out_format=FLAGS.out_format,
                                                                                  nbr_cores=FLAGS.nbr_cores)
        val_paths, val_slices, val_crops, val_aug_seeds = preprocess_data(val_paths, FLAGS.out_path, split="val",
                                                                          out_format=FLAGS.out_format,
                                                                          nbr_cores=FLAGS.nbr_cores)
        test_paths, test_slices, test_crops, test_aug_seeds = preprocess_data(test_paths, FLAGS.out_path, split='test',
                                                                              out_format=FLAGS.out_format,
                                                                              nbr_cores=FLAGS.nbr_cores)

        nbr_train_samples = sum(train_slices)
        nbr_val_samples = sum(val_slices)
        nbr_test_samples = sum(test_slices)

        data_info = {
            'train': {
                'size': nbr_train_samples,
                'paths': train_paths,
                'slices': train_slices,
                'crop_idx': train_crops
            },
            'val': {
                'size': nbr_val_samples,
                'paths': val_paths,
                'slices': val_slices,
                'crop_idx': val_crops
            },
            'test': {
                'size': nbr_test_samples,
                'paths': test_paths,
                'slices': test_slices,
                'crop_idx': test_crops
            }
        }

        aug_seeds = {
            'train': train_aug_seeds,
            'val': val_aug_seeds,
            'test': test_aug_seeds
        }

        with open(os.path.join(FLAGS.out_path, 'data_info.json'), 'w') as fp:
            json.dump(data_info, fp, indent=4)

        with open(os.path.join(FLAGS.out_path, 'aug_seed.json'), 'w') as fp:
            json.dump(aug_seeds, fp, indent=4)

        logging.info('Pre-processing finished')

        logging.info('{} training samples saved to {}'.format(nbr_train_samples, FLAGS.out_path))
        logging.info('{} validation samples saved to {}'.format(nbr_val_samples, FLAGS.out_path))
        logging.info('{} test samples saved to {}'.format(nbr_test_samples, FLAGS.out_path))
    else:
        logging.info('Preprocessed data already exists')


if __name__ == '__main__':
    flags.DEFINE_string(
        'in_path', default=DOWNLOADED_DATA_PATH,
        help='The path to the downloaded BRATS data.'
    )
    flags.DEFINE_string(
        'out_path', default=PROCESSED_DATA_PATH,
        help='Path where the processed data is saved.'
    )
    flags.DEFINE_enum(
        'out_format', default='tfrecord',
        enum_values=['numpy', 'tfrecord'],
        help='Choose the output format to be either .tfrecords- or .npy-files.'
    )
    flags.DEFINE_integer(
        'nbr_cores', default=1,
        help='The number of CPU-cores available for pre-processing.'
    )
    flags.DEFINE_float(
        'train_cut', default=0.8, lower_bound=0., upper_bound=1.,
        help='Part of the labeled data that should be used for training, '
             'the rest is used for validation'
    )
    flags.DEFINE_integer(
        'seed', default=42,
        help='Seed when splitting up training and validation sets.'
    )
    flags.DEFINE_integer(
        'nb_patients_per_file', default=1,
        help='Number of patients exported to each tfrecord or numpy file'
    )

    logging.set_verbosity(logging.INFO)
    app.run(main)
