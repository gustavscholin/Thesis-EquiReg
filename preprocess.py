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
EXAMPLE_DATA_PATH = 'data/raw_data/example_data'
PROCESSED_DATA_PATH = 'data/processed_data'

LABELED_DATA_FOLDER = 'MICCAI_BraTS_2019_Data_Training'

FLAGS = flags.FLAGS

os.environ['KMP_AFFINITY'] = 'disabled'


def download_brats19():
    # TODO
    pass


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def obtain_tfrecord_writer(out_path, shard_cnt):
    tfrecord_writer = tf.io.TFRecordWriter("{}.{:d}".format(out_path, shard_cnt))
    return tfrecord_writer


def save_tfrecord(example_list, out_path, shard_cnt):
    record_writer = obtain_tfrecord_writer(out_path, shard_cnt)
    for example in example_list:
        record_writer.write(example.SerializeToString())
    record_writer.close()
    logging.info("Saved {} examples to {}".format(len(example_list), out_path))


def save_npy(data_list, out_dir, shard_cnt):
    if data_list:
        data = np.concatenate(data_list, axis=0)
        out_path = '{}_{}.npy'.format(out_dir, shard_cnt)
        np.save(out_path, data)
        logging.info('Saved {} samples to {}'.format(data.shape[0], out_path))


def get_example(images, seg_masks):
    example_list = []
    for image, seg_mask in zip(images, seg_masks):
        feature = {
            'image': _float_feature(image.reshape(-1)),
            'seg_mask': _int_feature(seg_mask.reshape(-1))
        }
        example_list.append(tf.train.Example(features=tf.train.Features(feature=feature)))

    return example_list


def process_data(data_paths, out_path, split, out_format, nbr_cores=1):
    logging.info('Pre-processing {} data'.format(split))

    partial_preproc_one = partial(preproc_one, out_format=out_format)

    pool = Pool(processes=nbr_cores)
    shard_cnt = 0
    count = 0

    if out_format == 'tfrecord':
        example_lists = []
        file_name = os.path.join(out_path, "{}_data.tfrecord".format(split))

        for example_list in pool.imap(partial_preproc_one, data_paths):
            example_lists.append(example_list)
            if len(example_lists) == 5:
                examples = list(itertools.chain.from_iterable(example_lists))
                save_tfrecord(examples, file_name, shard_cnt)
                shard_cnt += 1
                example_lists = []
            count += len(example_list)

        examples = list(itertools.chain.from_iterable(example_lists))
        save_tfrecord(examples, file_name, shard_cnt)

    elif out_format == 'numpy':
        data_list = []
        images_file_path = os.path.join(out_path, '{}_images'.format(split))
        seg_masks_file_path = os.path.join(out_path, '{}_seg_masks'.format(split))

        for sample_tup in pool.imap(partial_preproc_one, data_paths):
            data_list.append(sample_tup)
            if len(data_list) == 3:
                save_npy([data[0] for data in data_list], images_file_path, shard_cnt)
                save_npy([data[1] for data in data_list], seg_masks_file_path, shard_cnt)
                shard_cnt += 1
                data_list = []
            count += len(sample_tup[0])

        save_npy([data[0] for data in data_list], images_file_path, shard_cnt)
        save_npy([data[1] for data in data_list], seg_masks_file_path, shard_cnt)

    return count


def get_data_paths(path):
    data_paths = []
    for root, dirnames, filenames in os.walk(path):
        if any('.nii.gz' in s for s in filenames):
            data_paths.append(root)
    return sorted(data_paths)


def preproc_one(path, out_format):
    logging.info('Processing {}'.format(path))
    mri_channels = []
    seg_masks = None
    for file in sorted(os.listdir(path)):
        if 'seg' not in file:
            mri_channels.append(sitk.ReadImage(os.path.join(path, file), sitk.sitkFloat32))
        else:
            seg_masks = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file), sitk.sitkInt64))
            seg_masks[np.where(seg_masks == 4)] = 3  # Moving all class 4 to class 3. Class 3 isn't used in BRATS 2019

    mri_channels, seg_masks = crop(mri_channels, seg_masks)

    for channel in range(4):
        mri_channels[channel] = bias_field_correction(mri_channels[channel])

    images = sitk.GetArrayFromImage(sitk.Compose(mri_channels))

    for channel in range(4):
        images[:, :, :, channel] = histogram_equalization(images[:, :, :, channel])

    if out_format == 'tfrecord':
        return get_example(images, seg_masks)

    elif out_format == 'numpy':
        return images, seg_masks


def crop(mri_channels, seg_masks):
    min_idx = 155
    max_idx = 0
    for mri_channel in mri_channels:
        mask = mri_channel != 0
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        bounding_box = label_shape_filter.GetBoundingBox(1)

        if bounding_box[2] < min_idx:
            min_idx = bounding_box[2]
        if bounding_box[5] + bounding_box[2] > max_idx:
            max_idx = bounding_box[5] + bounding_box[2]

    # Crop mri and seg-mask to 224x224 to fit tiramisu model
    return [mri_channel[8:232, 8:232, min_idx:max_idx] for mri_channel in mri_channels], \
           seg_masks[min_idx:max_idx, 8:232, 8:232]


def bias_field_correction(img):
    mask = (img != 0)
    img = sitk.Cast(img, sitk.sitkFloat32)
    return sitk.N4BiasFieldCorrection(img, mask)


def histogram_equalization(data):
    mask = (data != 0)
    data_flat = data[mask]

    n_bins = int(np.max(data_flat) + 1)

    histogram, bins = np.histogram(data_flat, n_bins, density=True)

    cdf = histogram.cumsum()
    cdf = cdf / cdf[-1]

    data[mask] = np.interp(data_flat, bins[:-1], cdf)

    return data


def main(argsv):
    data_path = os.path.join(FLAGS.in_path, LABELED_DATA_FOLDER)

    if not os.path.exists(data_path):
        logging.fatal('Input data not found')
        sys.exit()

    if not os.path.exists(FLAGS.out_path) or not os.listdir(FLAGS.out_path):
        logging.info('Starting pre-processing')

        if not os.path.exists(FLAGS.out_path):
            os.makedirs(FLAGS.out_path)

        data_paths = get_data_paths(data_path)
        random.shuffle(data_paths)
        nb_train_paths = int(len(data_paths) * FLAGS.train_cut)
        train_paths = data_paths[:nb_train_paths]
        val_paths = data_paths[nb_train_paths:]

        nbr_train_samples = process_data(train_paths, FLAGS.out_path, split="train",
                                         out_format=FLAGS.out_format, nbr_cores=FLAGS.nbr_cores)
        nbr_val_samples = process_data(val_paths, FLAGS.out_path, split="val",
                                       out_format=FLAGS.out_format, nbr_cores=FLAGS.nbr_cores)

        data_sizes = {
            'train_size': nbr_train_samples,
            'val_size': nbr_val_samples
        }
        with open(os.path.join(FLAGS.out_path, 'data_sizes.json'), 'w') as fp:
            json.dump(data_sizes, fp)

        logging.info('Pre-processing finished')

        logging.info('{} training samples saved to {}'.format(nbr_train_samples, FLAGS.out_path))
        logging.info('{} validation samples saved to {}'.format(nbr_val_samples, FLAGS.out_path))
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

    logging.set_verbosity(logging.INFO)
    app.run(main)
