"""
Data loading module.
"""
import math
import os
import random
import tensorflow as tf
import re

from typing import Callable
from absl import flags
from augmenters import unsup_img_aug, sup_aug

FLAGS = flags.FLAGS


def natural_sort(in_list: list) -> list:
    """
    Sort list of strings including integers in a human way.
    :param in_list: List of strings
    :return: Sorted list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    num_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(in_list, key=num_key)


def get_file_list(data_dir: str, split: str) -> list:
    """
    Get a list with paths to all .tfrecord-files.
    :param data_dir: Path to dir
    :param split: Whether it is training, validation or test data
    :return: List with .tfrecord-file paths
    """
    file_prefix = '{}_data.tfrecord'.format(split)
    file_list = natural_sort(tf.io.gfile.glob(os.path.join(data_dir, split, file_prefix + '*')))
    return file_list


def _postprocess_example(example: dict):
    """
    Convert tensor type, cast int64 into int32 and cast sparse to dense.
    :param example: Dict with tensors
    """
    for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
            val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
            val = tf.cast(val, dtype=tf.int32)
        example[key] = val


def get_dataset(type: str, data_dir: str, record_spec: dict,
                split: str, batch_size: int, data_size: int, cut: float, seed: int) -> tf.data.Dataset:
    """
    Get Tensorflow dataset.
    :param type: If the data is labeled or unlabeled
    :param data_dir: Path to .tfrecord-files
    :param record_spec: Specification for unpacking the .tfrecord-files
    :param split: Whether it is training, validation or test data
    :param batch_size: Number of samples in each batch
    :param data_size: Total number of samples
    :param cut: Percentage of the total number of samples included in the dataset
    :param seed: Seed for which samples to include in the dataset
    :return: A Tensorflow dataset
    """
    is_training = (split == "train")

    def parser(record: tf.Tensor) -> dict:
        """
        Parser to parse a single Tensorflow example. If training and the data is labeled
        it is weakly augmented. If training and the data is unlabeled one instance is weakly
        augmented and one instance is strongly augmented. A seed for the strong augmentation
        is saved. See thesis report for details.
        :param record: A single serialized example
        :return: Dict with tensors
        """
        # retrieve serialized example
        example = tf.io.parse_single_example(
            serialized=record,
            features=record_spec)
        # reshape image back to 2D shape
        if type == 'sup':
            example['image'] = tf.reshape(example['image'], [224, 224, 4])
            if 'test' not in split:
                example['seg_map'] = tf.reshape(example['seg_map'], [224, 224])
            if is_training:
                image, seg_map = tf.compat.v1.py_func(sup_aug, [example['image'], example['seg_map']],
                                                      (tf.float32, tf.int32))
                image.set_shape([224, 224, 4])
                seg_map.set_shape([224, 224])
                example = {
                    'image': image,
                    'seg_map': seg_map
                }
        elif type == 'unsup':
            ori_image = tf.reshape(example['image'], [224, 224, 4])
            ori_image, aug_image, seed_sq_ent = tf.compat.v1.py_func(unsup_img_aug, [ori_image],
                                                                     (tf.float32, tf.float32, tf.string))
            ori_image.set_shape([224, 224, 4])
            aug_image.set_shape([224, 224, 4])
            seed_sq_ent.set_shape([1])

            example = {
                'ori_image': ori_image,
                'aug_image': aug_image,
                'seed_sq_ent': seed_sq_ent
            }

        _postprocess_example(example)

        return example

    all_file_list = get_file_list(data_dir, split)

    if is_training:
        random.Random(seed).shuffle(all_file_list)
    if type == 'sup':
        cut_file_list = all_file_list[:math.ceil(len(all_file_list) * cut)]
    else:
        cut_file_list = all_file_list[-math.floor(len(all_file_list) * cut):]
    dataset = tf.data.Dataset.from_tensor_slices(cut_file_list)

    if not is_training:
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=1)
    else:
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=4)

    dataset = dataset.map(parser, num_parallel_calls=32)

    if is_training:
        buffer_size = int(data_size * (len(cut_file_list) / len(all_file_list)))
        if buffer_size > 3500:
            buffer_size = 3500
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(1)

    return dataset


def get_input_fn(
        data_dir: str, split: str, data_size: int, batch_size: int, sup_cut: float = 0.1,
        unsup_cut: float = 1.0, unsup_ratio: int = 0, shuffle_seed: int = None) -> Callable:
    """
    Get input data function.
    :param data_dir: Path to .tfrecord-files
    :param split: Whether it is training, validation or test data
    :param data_size: Total number of samples
    :param batch_size: Number of samples in each batch
    :param sup_cut: Percentage of the total number of labeled samples included
    :param unsup_cut: Percentage of the total number of unlabeled samples included
    :param unsup_ratio: Ratio between unlabeled and labeled samples in one batch
    :param shuffle_seed: Seed for which samples to include in the dataset
    :return: Input function
    """

    def input_fn() -> tf.data.Dataset:
        """
        Returns a Tensorflow dataset with both labeled and unlabeled samples in each batch.
        If training and the sample is labeled it is weakly augmented. If training and the sample
        is unlabeled one instance is weakly augmented and one instance is strongly augmented.
        :return: Tensorflow dataset
        """
        datasets = []

        record_spec = {
            'image': tf.io.FixedLenFeature([224 * 224 * 4], tf.float32)
        }
        if 'test' not in split:
            record_spec['seg_map'] = tf.io.FixedLenFeature([224 * 224], tf.int64)

        # Supervised data
        if sup_cut > 0:
            sup_dataset = get_dataset(
                type='sup',
                data_dir=data_dir,
                record_spec=record_spec,
                split=split,
                batch_size=batch_size,
                data_size=data_size,
                cut=sup_cut,
                seed=shuffle_seed
            )
            datasets.append(sup_dataset)

        if unsup_cut > 0 and unsup_ratio > 0:
            aug_dataset = get_dataset(
                type='unsup',
                data_dir=data_dir,
                record_spec=record_spec,
                split=split,
                batch_size=batch_size * unsup_ratio,
                data_size=data_size,
                cut=unsup_cut,
                seed=shuffle_seed
            )
            datasets.append(aug_dataset)

        def flatten_input(*features):
            result = {}
            for feature in features:
                for key in feature:
                    assert key not in result
                    result[key] = feature[key]
            return result

        if len(datasets) > 1:
            dataset = tf.data.Dataset.zip(tuple(datasets))
            dataset = dataset.map(flatten_input)
        else:
            dataset = datasets[0]

        return dataset

    return input_fn
