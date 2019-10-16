import sys
import numpy as np
import SimpleITK as sitk
import os
import tensorflow as tf
import itertools

from absl import flags, app, logging
from functools import partial
from multiprocessing import Pool


RAW_DATA_PATH = "data/raw_data"
DOWNLOADED_DATA_PATH = "data/raw_data/downloaded_data"

DOWNLOADED_TRAIN_FOLDER = "MICCAI_BraTS_2019_Data_Training"
DOWNLOADED_VAL_FOLDER = "MICCAI_BraTS_2019_Data_Validation"
PROCESSED_DATA_FOLDER = "processed_raw_data"

EXAMPLE_DATA_PATH = os.path.join(RAW_DATA_PATH, 'example_data')

DOWNLOADED_TRAIN_PATH = os.path.join(DOWNLOADED_DATA_PATH, DOWNLOADED_TRAIN_FOLDER)
DOWNLOADED_VAL_PATH = os.path.join(DOWNLOADED_DATA_PATH, DOWNLOADED_VAL_FOLDER)
PROCESSED_PATH = os.path.join(RAW_DATA_PATH, PROCESSED_DATA_FOLDER)
file_list = ["train_images.npy", "train_seg_maps.npy", "validation_images.npy"]

FLAGS = flags.FLAGS


def download_brats19():
    #TODO
    pass


def sup_unsup_split(unsup_ratio):
    #TODO
    pass


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


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


def get_example(images, segs=None):
    example_list = []
    if segs is not None:
        for image, seg in zip(images, segs):
            feature = {
                'image': _float_feature(image.reshape(-1)),
                'seg': _float_feature(seg.reshape(-1))
            }
            example_list.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    else:
        for image in images:
            feature = {
                'image': _float_feature(image.reshape(-1))
            }
            example_list.append(tf.train.Example(features=tf.train.Features(feature=feature)))

    return example_list


def process_data(in_path, out_path, labeled, split, out_format, nbr_cores=1):
    logging.info('Pre-processing {} data'.format(split))
    data_paths = get_data_paths(in_path)

    partial_preproc_one = partial(preproc_one, out_format=out_format)

    pool = Pool(processes=nbr_cores)
    shard_cnt = 0

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

        examples = list(itertools.chain.from_iterable(example_lists))
        save_tfrecord(examples, file_name, shard_cnt)

    elif out_format == 'numpy':
        data_list = []
        images_file_path = os.path.join(out_path, '{}_images'.format(split))
        segs_file_path = os.path.join(out_path, '{}_seg_maps'.format(split))

        for sample_tup in pool.imap(partial_preproc_one, data_paths):
            data_list.append(sample_tup)
            if len(data_list) == 3:
                save_npy([data[0] for data in data_list], images_file_path, shard_cnt)
                if labeled:
                    save_npy([data[1] for data in data_list], segs_file_path, shard_cnt)
                shard_cnt += 1
                data_list = []

        save_npy([data[0] for data in data_list], images_file_path, shard_cnt)
        if labeled:
            save_npy([data[1] for data in data_list], segs_file_path, shard_cnt)


def get_data_paths(path):
    data_paths =[]
    for root, dirnames, filenames in os.walk(path):
        if any('.nii.gz' in s for s in filenames):
            data_paths.append(root)
    return sorted(data_paths)


def preproc_one(path, out_format):
    logging.info('Processing {}'.format(path))
    mri_channels = []
    segs = None
    for file in sorted(os.listdir(path)):
        if 'seg' not in file:
            mri_channels.append(sitk.ReadImage(os.path.join(path, file), sitk.sitkFloat32))
        else:
            segs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file),
                                                                        sitk.sitkFloat32)), axis=-1)

    mri_channels, segs = crop(mri_channels, segs)

    for channel in range(4):
        # mri_channels[channel] = bias_field_correction_sitk(mri_channels[channel])
        pass

    images = sitk.GetArrayFromImage(sitk.Compose(mri_channels))

    for channel in range(4):
        images[:, :, :, channel] = histogram_equalization(images[:, :, :, channel])

    if out_format == 'tfrecord':
        return get_example(images, segs)

    elif out_format == 'numpy':
        return images, segs


def crop(mri_channels, segs=None):
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

    if segs is not None:
        return [mri_channel[:, :, min_idx:max_idx] for mri_channel in mri_channels], segs[min_idx:max_idx, :, :]
    else:
        return [mri_channel[:, :, min_idx:max_idx] for mri_channel in mri_channels], segs


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
    if not (os.path.exists(os.path.join(FLAGS.in_path, DOWNLOADED_TRAIN_FOLDER))
            or os.path.exists(os.path.join(FLAGS.in_path, DOWNLOADED_TRAIN_FOLDER))):
        logging.fatal('Input data not found')
        sys.exit()

    if not os.path.exists(FLAGS.out_path) or not all(file in file_list for file in os.listdir(FLAGS.out_path)): #TODO: fix this
        logging.info('Starting pre-processing')

        if not os.path.exists(FLAGS.out_path):
            os.makedirs(FLAGS.out_path)

        train_path = os.path.join(FLAGS.in_path, DOWNLOADED_TRAIN_FOLDER)
        val_path = os.path.join(FLAGS.in_path, DOWNLOADED_VAL_FOLDER)

        process_data(train_path, FLAGS.out_path, labeled=True, split="train",
                     out_format=FLAGS.out_format, nbr_cores=FLAGS.nbr_cores)
        process_data(val_path, FLAGS.out_path, labeled=False, split="val",
                     out_format=FLAGS.out_format, nbr_cores=FLAGS.nbr_cores)

        logging.info('Pre-processing finished')


if __name__ == '__main__':
    flags.DEFINE_string(
        'in_path', default=DOWNLOADED_DATA_PATH,
        help='The path to the downloaded BRATS data.'
    )
    flags.DEFINE_string(
        'out_path', default=PROCESSED_PATH,
        help='Path where the processed data is saved.'
    )
    flags.DEFINE_enum(
        'out_format', default='numpy',
        enum_values=['numpy', 'tfrecord'],
        help='Choose the output format to be either .tfrecords- or .npy-files.'
    )
    flags.DEFINE_integer(
        'nbr_cores', default=1,
        help='The number of CPU-cores available for pre-processing.'
    )

    logging.set_verbosity(logging.INFO)
    app.run(main)


