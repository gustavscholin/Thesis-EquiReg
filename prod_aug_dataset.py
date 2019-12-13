import glob
import json
import os
import re
from augmenters import unsup_img_aug
import tensorflow as tf
tf.enable_eager_execution()


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text
    num_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=num_key)


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
    print("Saved {} examples to {}".format(len(example_list), out_path))


def get_example_list(images):
    example_list = []
    for image in images:
        feature = {
            'image': _float_feature(image.reshape(-1))
        }
        example_list.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return example_list


seeds_dict = {}
with tf.gfile.Open(os.path.join('data/processed_data/data_info.json'), 'r') as fp:
    data_info = json.load(fp)

for split in ['val', 'test']:
    record_spec = {
        'image': tf.FixedLenFeature([224 * 224 * 4], tf.float32)
    }
    if not split == 'test':
        record_spec['seg_mask'] = tf.FixedLenFeature([224 * 224], tf.int64)

    def parser(record):
        # retrieve serialized example
        example = tf.parse_single_example(
            serialized=record,
            features=record_spec)
        return example

    os.makedirs('data/processed_data/{}_aug'.format(split))
    patient_idx = 0
    seeds_dict[split] = {}

    for file_path in natural_sort(glob.glob(os.path.join('data/processed_data', split, '*tfrecord*'))):

        patient_id = data_info[split]['paths'][patient_idx].split('/')[-1]
        patient_seeds = []

        dataset = tf.data.Dataset.from_tensor_slices([file_path])
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=1)
        dataset = dataset.map(parser, num_parallel_calls=32)
        iterator = dataset.make_one_shot_iterator()

        file_name = os.path.join('data/processed_data/{}_aug'.format(split), "{}_aug_data.tfrecord".format(split))

        aug_images = []

        for sample in iterator:
            image = tf.reshape(sample['image'], [224, 224, 4]).numpy()
            aug_image, ent = unsup_img_aug(image)
            aug_images.append(aug_image)
            patient_seeds.append(ent[0])

        example_list = get_example_list(aug_images)
        save_tfrecord(example_list, file_name, patient_idx)
        seeds_dict[split][patient_id] = patient_seeds
        print('Patient {} saved'.format(patient_idx))
        patient_idx += 1

with tf.gfile.Open(os.path.join('data/processed_data/aug_seed.json'), 'w') as fp:
    json.dump(seeds_dict, fp, indent=4)
