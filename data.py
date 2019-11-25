# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loading module of CIFAR && SVHN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import math
import os
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

from absl import flags
from augmenters import unsup_img_aug, sup_aug

FLAGS = flags.FLAGS


def get_file_list(data_dir, split):
    file_prefix = '{}_data.tfrecord'.format(split)
    file_list = tf.gfile.Glob(os.path.join(data_dir, split, file_prefix + '*'))
    return file_list


def _postprocess_example(example):
    """Convert tensor type for TPU, cast int64 into int32 and cast sparse to dense."""
    for key in list(example.keys()):
        val = example[key]
        if tf.keras.backend.is_sparse(val):
            val = tf.sparse.to_dense(val)
        if val.dtype == tf.int64:
            val = tf.to_int32(val)
        example[key] = val


def get_dataset(type, data_dir, record_spec,
                split, per_core_bsz, size, cut_size, seed):
    is_training = (split == "train")

    def parser(record):
        # retrieve serialized example
        example = tf.parse_single_example(
            serialized=record,
            features=record_spec)
        # reshape image back to 3D shape
        if type == 'sup':
            example['image'] = tf.reshape(example['image'], [224, 224, 4])
            if not split == 'test':
                example['seg_mask'] = tf.reshape(example['seg_mask'], [224, 224])
            if is_training:
                image, seg_mask = tf.py_func(sup_aug, [example['image'], example['seg_mask']], (tf.float32, tf.int32))
                image.set_shape([224, 224, 4])
                seg_mask.set_shape([224, 224])
                example = {
                    'image': image,
                    'seg_mask': seg_mask
                }
        elif type == 'unsup':
            ori_image = tf.reshape(example['image'], [224, 224, 4])
            aug_image, seed_sq_ent = tf.py_func(unsup_img_aug, [ori_image], (tf.float32, tf.string))
            aug_image.set_shape([224, 224, 4])
            seed_sq_ent.set_shape([1])
            # TODO: Augmentation on original image?
            example = {
                'ori_image': ori_image,
                'aug_image': aug_image,
                'seed_sq_ent': seed_sq_ent
            }

        _postprocess_example(example)

        return example

    all_file_list = get_file_list(data_dir, split)
    dataset = tf.data.Dataset.from_tensor_slices(all_file_list)

    if not is_training:
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=1)
    else:
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=4)
        
    dataset = dataset.map(parser, num_parallel_calls=32)

    if is_training:
        buffer_size = int(size * 0.1)
        if cut_size < size:
            dataset = dataset.shuffle(buffer_size, seed)
            if type == 'sup':
                dataset = dataset.take(cut_size)
            else:
                dataset = dataset.skip(size - cut_size)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.repeat()
    dataset = dataset.batch(per_core_bsz, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def get_input_fn(
        data_dir, split, data_info, batch_size, sup_cut=0.1,
        unsup_cut=1.0, unsup_ratio=0, shuffle_seed=None):
    def input_fn(params):

        size = data_info[split]['size']
        sup_cut_size = int(math.ceil(sup_cut * size))
        unsup_cut_size = int(math.floor(unsup_cut * size))

        datasets = []

        record_spec = {
            "image": tf.FixedLenFeature([224 * 224 * 4], tf.float32),
            "seg_mask": tf.FixedLenFeature([224 * 224], tf.int64)
        }

        # Supervised data
        if sup_cut_size > 0:
            sup_dataset = get_dataset(
                type='sup',
                data_dir=data_dir,
                record_spec=record_spec,
                split=split,
                per_core_bsz=batch_size,
                size=size,
                cut_size=sup_cut_size,
                seed=shuffle_seed
            )
            datasets.append(sup_dataset)

        if unsup_cut_size > 0 and unsup_ratio > 0:
            aug_dataset = get_dataset(
                type='unsup',
                data_dir=data_dir,
                record_spec=record_spec,
                split=split,
                per_core_bsz=batch_size * unsup_ratio,
                size=size,
                cut_size=unsup_cut_size,
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
