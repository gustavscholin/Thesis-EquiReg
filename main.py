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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import json
import numpy as np
import tensorflow as tf
import data
import utils
import SimpleITK as sitk

from absl import flags
from models.tiramisu import DenseTiramisu
from augmenters import unsup_logits_aug
from prediction_dice import calc_and_export_dice

os.environ['KMP_AFFINITY'] = 'disabled'

# UDA config:
flags.DEFINE_float(
    "sup_cut", default=0.1, lower_bound=0.0, upper_bound=1.0,
    help="How much supervised training data to use")
flags.DEFINE_float(
    "unsup_cut", default=0.9, lower_bound=0.0, upper_bound=1.0,
    help="How much unsupervised training data to use")
flags.DEFINE_integer(
    "unsup_ratio", default=5,
    help="The ratio between batch size of unlabeled data and labeled data, "
         "i.e., unsup_ratio * train_batch_size is the batch_size for unlabeled data."
         "Do not use the unsupervised objective if set to 0.")
flags.DEFINE_integer(
    'shuffle_seed', default=None,
    help='Seed for shuffling the training dataset.'
)
flags.DEFINE_enum(
    "tsa", "",
    enum_values=["", "linear_schedule", "log_schedule", "exp_schedule", "soft"],
    help="anneal schedule of training signal annealing. "
         "tsa='' means not using TSA. See the paper for other schedules.")
flags.DEFINE_float(
    "uda_confidence_thresh", default=-1,
    help="The threshold on predicted probability on unsupervised data. If set,"
         "UDA loss will only be calculated on unlabeled examples whose largest"
         "probability is larger than the threshold")
flags.DEFINE_float(
    "uda_softmax_temp", -1,
    help="The temperature of the Softmax when making prediction on unlabeled"
         "examples. -1 means to use normal Softmax")
flags.DEFINE_float(
    "ent_min_coeff", default=0,
    help="")
flags.DEFINE_float(
    "unsup_coeff", default=1.0,
    help="The coefficient on the UDA loss. "
         "setting unsup_coeff to 1 works for most settings. "
         "When you have extermely few samples, consider increasing unsup_coeff")
flags.DEFINE_bool(
    'unsup_crop', default=False,
    help='Whether to only calculate unsup loss from non-zero input pixels or not.'
)

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string(
    "data_dir", default=None,
    help="Path to data directory containing `*.tfrecords`.")
flags.DEFINE_string(
    "model_dir", default=None,
    help="model dir of the saved checkpoints.")
flags.DEFINE_bool(
    "do_train", default=False,
    help="Whether to run training.")
flags.DEFINE_bool(
    "do_eval", default=False,
    help="Whether to run eval on the test set.")
flags.DEFINE_bool(
    "do_eval_along_training", default=True,
    help="Whether to run eval on the test set during training. "
         "This is only used to debug.")
flags.DEFINE_bool(
    'do_predict', default=True,
    help='Whether to run predict.'
)
flags.DEFINE_enum(
    'pred_dataset', default='val',
    enum_values=['test', 'val'],
    help='Whether to predict the test or the validation dataset.'
)
flags.DEFINE_string(
    'pred_ckpt', default='best',
    help='Checkpoint export to run prediction on. If best, prediction '
         'on the checkpoint with the lowest eval loss is done. Otherwise '
         'export dir name must be specified.'
)
flags.DEFINE_bool(
    'plot_train_images', default=False,
    help='Whether to plot eval images to tensorboard during training.'
)
flags.DEFINE_bool(
    'plot_eval_images', default=False,
    help='Whether to plot train images to tensorboard during training.'
)
flags.DEFINE_bool(
    "verbose", default=False,
    help="Whether to print additional information.")

# Training config
flags.DEFINE_integer(
    "train_batch_size", default=4,
    help="Size of train batch.")
flags.DEFINE_integer(
    "eval_batch_size", default=8,
    help="Size of evalation batch.")
flags.DEFINE_integer(
    "pred_batch_size", default=8,
    help="Size of prediction batch.")
flags.DEFINE_integer(
    "train_steps", default=100000,
    help="Total number of training steps.")
flags.DEFINE_integer(
    "iterations", default=500,
    help="Number of iterations per repeat loop.")
flags.DEFINE_integer(
    "save_steps", default=500,
    help="number of steps for model checkpointing.")
flags.DEFINE_integer(
    "max_save", default=1,
    help="Maximum number of checkpoints to save.")
flags.DEFINE_integer(
    'early_stop_steps', default=10000,
    help='Training will stop if the eval loss has not '
         'decreased in early_stop_steps number of steps. '
         'If -1, early stopping is disabled.'
)

# Model config
flags.DEFINE_enum(
    "model_name", default="tiramisu",
    enum_values=['tiramisu', 'unet'],
    help="Name of the model")
flags.DEFINE_integer(
    "num_classes", default=4,
    help="Number of categories for classification.")

# Optimization config
flags.DEFINE_float(
    "learning_rate", default=0.003,
    help="Maximum learning rate.")
flags.DEFINE_bool(
    'const_lr', default=True,
    help='Whether the learning rate will be constant during training or not.'
)
flags.DEFINE_float(
    "weight_decay_rate", default=1e-4,
    help="Weight decay rate.")
flags.DEFINE_float(
    "min_lr_ratio", default=0.004,
    help="Minimum ratio learning rate.")
flags.DEFINE_integer(
    "warmup_steps", default=20000,
    help="Number of steps for linear lr warmup.")

FLAGS = tf.flags.FLAGS

arg_scope = tf.contrib.framework.arg_scope


def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    step_ratio = tf.to_float(global_step) / tf.to_float(num_train_steps)
    if schedule == "linear_schedule":
        coeff = step_ratio
    elif schedule == "exp_schedule":
        scale = 5
        # [exp(-5), exp(0)] = [1e-2, 1]
        coeff = tf.exp((step_ratio - 1) * scale)
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        coeff = 1 - tf.exp((-step_ratio) * scale)
    return coeff * (end - start) + start


def _kl_divergence_with_logits(p_logits, q_logits):
    p = tf.nn.softmax(p_logits)
    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)

    kl = tf.reduce_sum(p * (log_p - log_q), -1)
    return kl


def anneal_sup_loss(sup_logits, sup_labels, sup_loss, global_step, training_summaries):
    one_hot_labels = tf.one_hot(
        sup_labels, depth=FLAGS.num_classes, dtype=tf.float32)
    sup_probs = tf.nn.softmax(sup_logits, axis=-1)
    correct_label_probs = tf.reduce_sum(
        one_hot_labels * sup_probs, axis=-1)

    if FLAGS.tsa == 'soft':
        loss_mask = 1 - tf.cast(correct_label_probs, tf.float32)
        loss_mask = tf.stop_gradient(loss_mask)

        sup_loss = sup_loss * loss_mask
        avg_sup_loss = tf.reduce_mean(tf.reduce_mean(sup_loss, axis=(-1, -2)))
    else:
        tsa_start = 1. / FLAGS.num_classes
        eff_train_prob_threshold = get_tsa_threshold(
            FLAGS.tsa, global_step, FLAGS.train_steps,
            tsa_start, end=1)

        larger_than_threshold = tf.greater(
            correct_label_probs, eff_train_prob_threshold)
        loss_mask = 1 - tf.cast(larger_than_threshold, tf.float32)
        loss_mask = tf.stop_gradient(loss_mask)

        training_summaries.append(
            tf.summary.scalar('sup/sup_trained_ratio', tf.reduce_mean(loss_mask)))
        training_summaries.append(
            tf.summary.scalar('sup/eff_train_prob_threshold', eff_train_prob_threshold))

        sup_loss = sup_loss * loss_mask
        avg_sup_loss = tf.reduce_mean((tf.reduce_sum(sup_loss, axis=(-1, -2)) /
                                       tf.maximum(tf.reduce_sum(loss_mask, axis=(-1, -2)), 1)))

    return sup_loss, avg_sup_loss


def get_ent(logits, return_mean=True):
    log_prob = tf.nn.log_softmax(logits, axis=-1)
    prob = tf.exp(log_prob)
    ent = tf.reduce_sum(-prob * log_prob, axis=-1)
    if return_mean:
        ent = tf.reduce_mean(ent)
    return ent


@tf.function
def class_convert(true_masks, pred_masks, brats_class):
    brats_class_mapping = {
        'whole': tf.constant([0, 1, 1, 1], dtype='float32'),
        'core': tf.constant([0, 1, 0, 1], dtype='float32'),
        'enhancing': tf.constant([0, 0, 0, 1], dtype='float32')
    }
    true_masks = tf.one_hot(true_masks, depth=FLAGS.num_classes)
    pred_masks = tf.one_hot(pred_masks, depth=FLAGS.num_classes)

    bool_true_masks = tf.tensordot(tf.cast(true_masks, tf.float32), brats_class_mapping[brats_class], axes=1)
    bool_pred_masks = tf.tensordot(tf.cast(pred_masks, tf.float32), brats_class_mapping[brats_class], axes=1)

    return tf.tuple([tf.cast(bool_true_masks, tf.int32), tf.cast(bool_pred_masks, tf.int32)])


@tf.function
def dice_coef(true, pred):
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, 1), tf.equal(pred, 1)), tf.float32), axis=(-1, -2))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, 0), tf.equal(pred, 1)), tf.float32), axis=(-1, -2))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(true, 1), tf.equal(pred, 0)), tf.float32), axis=(-1, -2))

    dice_coefs = tf.math.divide_no_nan(2 * tp, 2 * tp + fp + fn) + tf.cast(tf.equal(tp + fp + fn, 0), tf.float32)

    return dice_coefs


def get_model_fn():
    def model_fn(features, labels, mode, params):
        # tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)
        # model = DenseNetFCN((224, 224, 4), classes=FLAGS.num_classes)
        model = DenseTiramisu(growth_k=16, layers_per_block=[4, 5, 7, 10, 12, 15], num_classes=FLAGS.num_classes)
        # if not (mode == tf.estimator.ModeKeys.PREDICT and FLAGS.pred_dataset == 'test'):

        #### Configuring the optimizer
        global_step = tf.train.get_global_step()
        metric_dict = {}
        training_summaries = []
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if FLAGS.unsup_ratio > 0 and is_training:
            all_images = tf.concat([features["image"],
                                    features["ori_image"],
                                    features["aug_image"]], 0)
        else:
            all_images = features["image"]

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            all_logits = model.model(all_images, is_training)

        sup_bsz = tf.shape(features["image"])[0]
        sup_logits = all_logits[:sup_bsz]

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = tf.argmax(sup_logits, axis=-1, output_type=tf.int32)
            output = {
                'prediction': predictions,
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=output,
                export_outputs={
                    'output': tf.estimator.export.PredictOutput(output)
                })

        sup_masks = features['seg_mask']
        sup_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sup_masks,
            logits=sup_logits)
        sup_prob = tf.nn.softmax(sup_logits, axis=-1)
        training_summaries.append(tf.summary.scalar('sup/pred_prob', tf.reduce_mean(tf.reduce_mean(
            tf.reduce_max(sup_prob, axis=-1), axis=(-1, -2)))))
        if FLAGS.tsa and is_training:
            # TODO: Implement TSA
            sup_loss, avg_sup_loss = anneal_sup_loss(sup_logits, sup_masks, sup_loss,
                                                     global_step, training_summaries)
        else:
            avg_sup_loss = tf.reduce_mean(tf.reduce_mean(sup_loss, axis=(-1, -2)))
        total_loss = avg_sup_loss
        training_summaries.append(tf.summary.scalar('sup/loss', avg_sup_loss))

        if FLAGS.unsup_ratio > 0 and is_training:
            aug_bsz = tf.shape(features["ori_image"])[0]

            ori_logits = all_logits[sup_bsz: sup_bsz + aug_bsz]
            aug_logits = all_logits[sup_bsz + aug_bsz:]
            if FLAGS.uda_softmax_temp != -1:
                # TODO: Find out if this is needed
                ori_logits_tgt = ori_logits / FLAGS.uda_softmax_temp
            else:
                ori_logits_tgt = ori_logits
            ori_logits_aug = tf.py_func(unsup_logits_aug, [ori_logits_tgt, features['seed_sq_ent']], tf.float32,
                                        stateful=False)
            ori_logits_aug.set_shape(ori_logits_tgt.shape)
            ori_prob = tf.nn.softmax(ori_logits_aug, axis=-1)
            aug_prob = tf.nn.softmax(aug_logits, axis=-1)

            training_summaries.append(tf.summary.scalar('unsup/ori_prob', tf.reduce_mean(tf.reduce_mean(
                tf.reduce_max(ori_prob, axis=-1), axis=(-1, -2)))))
            training_summaries.append(tf.summary.scalar('unsup/aug_prob', tf.reduce_mean(tf.reduce_mean(
                tf.reduce_max(aug_prob, axis=-1), axis=(-1, -2)))))

            aug_loss = _kl_divergence_with_logits(
                p_logits=tf.stop_gradient(ori_logits_aug),
                q_logits=aug_logits)

            # if FLAGS.uda_confidence_thresh != -1:
            #     ori_prob = tf.nn.softmax(ori_logits, axis=-1)
            #     largest_prob = tf.reduce_max(ori_prob, axis=-1)
            #     loss_mask = tf.cast(tf.greater(
            #         largest_prob, FLAGS.uda_confidence_thresh), tf.float32)
            #     metric_dict["unsup/high_prob_ratio"] = tf.reduce_mean(loss_mask)
            #     loss_mask = tf.stop_gradient(loss_mask)
            #     aug_loss = aug_loss * loss_mask
            #     metric_dict["unsup/high_prob_loss"] = tf.reduce_mean(aug_loss)

            # if FLAGS.ent_min_coeff > 0:
            #     ent_min_coeff = FLAGS.ent_min_coeff
            #     metric_dict["unsup/ent_min_coeff"] = ent_min_coeff
            #     per_example_ent = get_ent(ori_logits)
            #     ent_min_loss = tf.reduce_mean(per_example_ent)
            #     total_loss = total_loss + ent_min_coeff * ent_min_loss

            if FLAGS.unsup_crop:
                aug_image_sum = tf.reduce_sum(features['aug_image'], axis=-1)
                loss_mask = tf.cast(tf.greater(aug_image_sum, tf.zeros(aug_image_sum.shape)), tf.float32)
                loss_mask = tf.stop_gradient(loss_mask)

                aug_loss = aug_loss * loss_mask
                avg_unsup_loss = tf.reduce_mean((tf.reduce_sum(aug_loss, axis=(-1, -2)) /
                                                 tf.maximum(tf.reduce_sum(loss_mask, axis=(-1, -2)), 1)))
            else:
                avg_unsup_loss = tf.reduce_mean(tf.reduce_mean(aug_loss, axis=(-1, -2)))

            total_loss += FLAGS.unsup_coeff * avg_unsup_loss
            training_summaries.append(tf.summary.scalar('unsup/loss', avg_unsup_loss))

        # total_loss = utils.decay_weights(
        #    total_loss,
        #    FLAGS.weight_decay_rate)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info("#params: {}".format(num_params))

        if FLAGS.verbose:
            format_str = "{{:<{0}s}}\t{{}}".format(
                max([len(v.name) for v in tf.trainable_variables()]))
            for v in tf.trainable_variables():
                tf.logging.info(format_str.format(v.name, v.get_shape()))

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = tf.argmax(sup_logits, axis=-1, output_type=tf.int32)

            loss = tf.metrics.mean(tf.reduce_mean(sup_loss, axis=(-1, -2)))

            brats_classes = ['whole', 'core', 'enhancing']
            dice_scores = {}
            for brats_class in brats_classes:
                true, pred = class_convert(sup_masks, predictions, brats_class)
                dice_scores[brats_class] = tf.metrics.mean(dice_coef(true, pred))

            eval_metrics = {
                "eval/classify_loss": loss,
                "eval/classify_whole_dice": dice_scores['whole'],
                "eval/classify_core_dice": dice_scores['core'],
                "eval/classify_enhancing_dice": dice_scores['enhancing']
            }

            if FLAGS.plot_eval_images:
                tf.summary.image('input', tf.expand_dims(all_images[..., 0], -1), 2)
                tf.summary.image('gt_mask', tf.cast(tf.expand_dims(sup_masks, -1), tf.float32), 2)
                tf.summary.image('pred_mask', tf.cast(tf.expand_dims(predictions, -1), tf.float32), 2)

            eval_summary_hook = tf.train.SummarySaverHook(
                save_secs=120,
                output_dir=FLAGS.model_dir,
                summary_op=tf.summary.merge_all())

            #### Constucting evaluation TPUEstimatorSpec.
            eval_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=avg_sup_loss,
                eval_metric_ops=eval_metrics,
                evaluation_hooks=[eval_summary_hook])

            return eval_spec

        if FLAGS.const_lr:
            learning_rate = tf.Variable(FLAGS.learning_rate)
        else:
            # increase the learning rate linearly
            if FLAGS.warmup_steps > 0:
                warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                            * FLAGS.learning_rate
            else:
                warmup_lr = 0.0

            # decay the learning rate using the cosine schedule
            decay_lr = tf.train.cosine_decay(
                FLAGS.learning_rate,
                global_step=global_step - FLAGS.warmup_steps,
                decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
                alpha=FLAGS.min_lr_ratio)

            learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                                     warmup_lr, decay_lr)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        grads_and_vars = optimizer.compute_gradients(total_loss)
        gradients, variables = zip(*grads_and_vars)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(
                zip(gradients, variables), global_step=tf.train.get_global_step())

        #### Creating training logging hook
        # compute accuracy
        sup_pred = tf.argmax(sup_logits, axis=-1, output_type=sup_masks.dtype)
        is_correct = tf.to_float(tf.equal(sup_pred, sup_masks))
        acc = tf.reduce_mean(is_correct)
        metric_dict["sup/sup_loss"] = avg_sup_loss
        metric_dict["training/loss"] = total_loss
        metric_dict["sup/acc"] = acc
        metric_dict["training/lr"] = learning_rate
        metric_dict["training/step"] = global_step

        log_info = ("step [{training/step}] lr {training/lr:.6f} "
                    "loss {training/loss:.4f} "
                    "sup/acc {sup/acc:.4f} sup/loss {sup/sup_loss:.6f} ")
        if FLAGS.unsup_ratio > 0:
            metric_dict["unsup/loss"] = avg_unsup_loss
            log_info += "unsup/loss {unsup/loss:.6f} "
        formatter = lambda kwargs: log_info.format(**kwargs)
        logging_hook = tf.train.LoggingTensorHook(
            tensors=metric_dict,
            every_n_iter=FLAGS.iterations,
            formatter=formatter)

        if FLAGS.unsup_ratio > 0 and FLAGS.plot_train_images:
            training_summaries.append(
                tf.summary.image('ori_image', tf.expand_dims(features['ori_image'][..., 0], -1), 1))
            training_summaries.append(
                tf.summary.image('aug_image', tf.expand_dims(features['aug_image'][..., 0], -1), 1))
            training_summaries.append(tf.summary.image('ori_mask', tf.cast(
                tf.expand_dims(tf.argmax(ori_logits_aug, axis=-1, output_type=tf.int32), -1),
                tf.float32), 1))
            training_summaries.append(tf.summary.image('aug_mask', tf.cast(
                tf.expand_dims(tf.argmax(aug_logits, axis=-1, output_type=tf.int32), -1),
                tf.float32), 1))

            training_summary_hook = tf.train.SummarySaverHook(
                save_steps=FLAGS.iterations,
                output_dir=FLAGS.model_dir,
                summary_op=training_summaries
            )
            training_hooks = [logging_hook, training_summary_hook]
        else:
            training_hooks = [logging_hook]

        #### Constucting training TPUEstimatorSpec.
        train_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op,
            training_hooks=training_hooks)

        return train_spec

    return model_fn


def train():
    # Create input function
    with tf.gfile.Open(os.path.join(FLAGS.data_dir, 'data_info.json'), 'r') as fp:
        data_info = json.load(fp)
    if FLAGS.unsup_ratio == 0:
        FLAGS.unsup_cut = 0.0

    train_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split="train",
        data_info=data_info,
        batch_size=FLAGS.train_batch_size,
        sup_cut=FLAGS.sup_cut,
        unsup_cut=FLAGS.unsup_cut,
        unsup_ratio=FLAGS.unsup_ratio,
    )

    eval_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split="val",
        data_info=data_info,
        batch_size=FLAGS.eval_batch_size,
        sup_cut=1.0,
        unsup_cut=0.0,
        unsup_ratio=0
    )

    test_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split=FLAGS.pred_dataset,
        data_info=data_info,
        batch_size=FLAGS.pred_batch_size,
        sup_cut=1.0,
        unsup_cut=0.0,
        unsup_ratio=0
    )

    eval_size = data_info['val']['size']
    eval_steps = eval_size // FLAGS.eval_batch_size

    # Get model function
    model_fn = get_model_fn()
    estimator = utils.get_estimator(FLAGS, model_fn)

    # Training
    if FLAGS.do_eval_along_training:
        tf.logging.info("***** Running training & evaluation *****")
        tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Unsupervised batch size = %d",
                        FLAGS.train_batch_size * FLAGS.unsup_ratio)
        tf.logging.info("  Num train steps = %d", FLAGS.train_steps)

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            {'image': tf.placeholder(tf.float32, [None, 224, 224, 4], name='input_images')})

        best_exporter = tf.estimator.BestExporter(
            name="best_exporter",
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=1)

        latest_exporter = tf.estimator.LatestExporter(
            name="latest_exporter",
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=None)

        hooks = []

        if FLAGS.early_stop_steps != -1:
            # Hook to stop training if loss does not decrease in over 10000 steps.
            hooks.append(tf.estimator.experimental.stop_if_no_decrease_hook(estimator, "loss", 10000))

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_steps, hooks=hooks)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps,
                                          exporters=[best_exporter, latest_exporter], start_delay_secs=0,
                                          throttle_secs=10)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        if FLAGS.do_train:
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Unsupervised batch size = %d",
                            FLAGS.train_batch_size * FLAGS.unsup_ratio)
            estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
        if FLAGS.do_eval:
            tf.logging.info("***** Running evaluation *****")
            results = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            tf.logging.info(">> Results:")
            for key in results.keys():
                tf.logging.info("  %s = %s", key, str(results[key]))
                results[key] = results[key].item()
            acc = results["eval/classify_accuracy"]
            with tf.gfile.Open("{}/results.txt".format(FLAGS.model_dir), "w") as ouf:
                ouf.write(str(acc))
    if FLAGS.do_predict:
        tf.enable_eager_execution()

        tf.logging.info('***** Running prediction *****')

        if FLAGS.pred_ckpt == 'best':
            out_path = os.path.join(FLAGS.model_dir, 'best_{}_prediction'.format(FLAGS.pred_dataset))
            export_dir = sorted(glob.glob(os.path.join(FLAGS.model_dir, 'export/best_exporter/*')))[-1]
        else:
            out_path = os.path.join(FLAGS.model_dir, '{}_{}_prediction'.format(FLAGS.pred_ckpt, FLAGS.pred_dataset))
            export_dir = os.path.join(FLAGS.model_dir, 'export/latest_exporter/{}'.format(FLAGS.pred_ckpt))

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        model = tf.saved_model.load_v2(export_dir)
        predict = model.signatures["serving_default"]

        dataset = test_input_fn()
        iterator = dataset.make_one_shot_iterator()

        preds = []

        example_cnt = 1
        patient_cnt = 0

        for sample in iterator:
            prediction_batch = predict(image=sample['image'])['prediction'].numpy()
            for i in range(prediction_batch.shape[0]):
                prediction = prediction_batch[i, ...]

                prediction[np.where(prediction == 3)] = 4
                preds.append(np.pad(prediction, ((8, 8), (8, 8))))

                if example_cnt == data_info[FLAGS.pred_dataset]['slices'][patient_cnt]:
                    patient_id = data_info[FLAGS.pred_dataset]['paths'][patient_cnt].split('/')[-1]

                    # Make the prediction dims (155,240,240) again for BRATS evaluation
                    below_padding = np.zeros((data_info[FLAGS.pred_dataset]['crop_idx'][patient_cnt][0], 240, 240))
                    above_padding = np.zeros(
                        (155 - data_info[FLAGS.pred_dataset]['crop_idx'][patient_cnt][1], 240, 240))
                    fill_dim_img = np.concatenate([below_padding, np.stack(preds), above_padding])

                    nii_img = sitk.GetImageFromArray(fill_dim_img)
                    sitk.WriteImage(nii_img, os.path.join(out_path, '{}.nii.gz'.format(patient_id)))

                    tf.logging.info('Exported patient {}'.format(patient_id))
                    example_cnt = 1
                    patient_cnt += 1
                    preds = []
                else:
                    example_cnt += 1

        if FLAGS.pred_dataset == 'val':
            tf.logging.info('Calculating Dice scores')
            calc_and_export_dice(out_path)


def main(_):
    if FLAGS.do_train:
        tf.gfile.MakeDirs(FLAGS.model_dir)
        flags_dict = tf.app.flags.FLAGS.flag_values_dict()
        with tf.gfile.Open(os.path.join(FLAGS.model_dir, "FLAGS.json"), "w") as ouf:
            json.dump(flags_dict, ouf)

    train()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
