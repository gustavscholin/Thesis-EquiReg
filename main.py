"""
Main script for training and evaluating baseline and EquiReg-models.
"""
import glob
import itertools
import os
import json
import numpy as np
import tensorflow as tf
import best_ckpt_copier as best_ckpt_copier
import data as data
import utils as utils
import SimpleITK as sitk

from absl import flags
from tiramisu import DenseTiramisu
from augmenters import unsup_logits_aug, data_aug
from utils import calc_and_export_standard_dice, calc_and_export_equivariance_dice
from typing import Callable

# EquiReg config:
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
         "tsa='' means not using TSA. "
         "Not used in the thesis.")
flags.DEFINE_float(
    "softmax_temp", -1,
    help="The temperature of the Softmax when making prediction on unlabeled"
         "examples. -1 means to use normal Softmax. "
         "Not used in the thesis.")
flags.DEFINE_float(
    "unsup_coeff", default=1.0,
    help="The coefficient on the EquiReg loss. "
         "Setting unsup_coeff to 1 works for most settings.")
flags.DEFINE_bool(
    'unsup_crop', default=True,
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
    "do_eval_along_training", default=True,
    help="Whether to run eval on the test set during training.")
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
    'plot_train_images', default=True,
    help='Whether to plot eval images to tensorboard during training.'
)
flags.DEFINE_bool(
    'plot_eval_images', default=True,
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
    "train_summary_steps", default=500,
    help="Number of steps for each train summary.")
flags.DEFINE_integer(
    "save_steps", default=500,
    help="Number of steps for model checkpointing.")
flags.DEFINE_integer(
    "max_save", default=1,
    help="Maximum number of checkpoints to save.")
flags.DEFINE_integer(
    'early_stop_steps', default=10000,
    help='Training will stop if the eval loss has not '
         'decreased in early_stop_steps number of steps. '
         'If -1, early stopping is disabled.')
flags.DEFINE_integer(
    'min_step', default=0,
    help='Training will not stop earlier than this step.')

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
    "learning_rate", default=0.001,
    help="Maximum learning rate.")
flags.DEFINE_bool(
    'cos_lr_dec', default=True,
    help='Whether to do cosine learning rate decay.'
)
flags.DEFINE_bool(
    'exp_lr_decay', default=False,
    help='Whether to do exponential learning rate decay.'
)
flags.DEFINE_bool(
    'do_weight_decay', default=False,
    help='Whether to do weight decay during training. '
         'Not used in thesis.'
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

FLAGS = flags.FLAGS


def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    step_ratio = tf.cast(global_step, dtype=tf.float32) / tf.cast(num_train_steps, dtype=tf.float32)
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


def anneal_sup_loss(sup_logits, sup_labels, sup_loss, global_step, training_summaries):
    one_hot_labels = tf.one_hot(
        sup_labels, depth=FLAGS.num_classes, dtype=tf.float32)
    sup_probs = tf.nn.softmax(sup_logits, axis=-1)
    correct_label_probs = tf.reduce_sum(
        input_tensor=one_hot_labels * sup_probs, axis=-1)

    if FLAGS.tsa == 'soft':
        loss_mask = 1 - tf.cast(correct_label_probs, tf.float32)
        loss_mask = tf.stop_gradient(loss_mask)

        sup_loss = sup_loss * loss_mask
        avg_sup_loss = tf.reduce_mean(input_tensor=tf.reduce_mean(input_tensor=sup_loss, axis=(-1, -2)))
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
            tf.compat.v1.summary.scalar('sup/sup_trained_ratio', tf.reduce_mean(input_tensor=loss_mask)))
        training_summaries.append(
            tf.compat.v1.summary.scalar('sup/eff_train_prob_threshold', eff_train_prob_threshold))

        sup_loss = sup_loss * loss_mask
        avg_sup_loss = tf.reduce_mean(input_tensor=(tf.reduce_sum(input_tensor=sup_loss, axis=(-1, -2)) /
                                                    tf.maximum(tf.reduce_sum(input_tensor=loss_mask, axis=(-1, -2)),
                                                               1)))

    return sup_loss, avg_sup_loss


def _kl_divergence_with_logits(p_logits: tf.Tensor, q_logits: tf.Tensor) -> tf.Tensor:
    """
    Calculate KL-divergence from probability distribution q to probability distribution p.
    :param p_logits: To logits
    :param q_logits: From logits
    :return: The Kl-divergence
    """
    p = tf.nn.softmax(p_logits)
    log_p = tf.nn.log_softmax(p_logits)
    log_q = tf.nn.log_softmax(q_logits)

    kl = tf.reduce_sum(input_tensor=p * (log_p - log_q), axis=-1)
    return kl


@tf.function
def class_convert(true_masks: tf.Tensor, pred_masks: tf.Tensor, brats_class: str) -> list:
    """
    Converts ground truth and predictions tensors to binary tensors.
    The binary conversion is defined by the BraTS binary problem.
    :param true_masks: Ground truth tensor
    :param pred_masks: Prediction tensor
    :param brats_class: Binary problem
    :return: Binary ground truth and prediction
    """
    brats_class_mapping = {
        'whole': tf.constant([0, 1, 1, 1], dtype='float32'),
        'core': tf.constant([0, 1, 0, 1], dtype='float32'),
        'enhancing': tf.constant([0, 0, 0, 1], dtype='float32')
    }
    true_masks = tf.one_hot(true_masks, depth=FLAGS.num_classes)
    pred_masks = tf.one_hot(pred_masks, depth=FLAGS.num_classes)

    bool_true_masks = tf.tensordot(tf.cast(true_masks, tf.float32), brats_class_mapping[brats_class], axes=1)
    bool_pred_masks = tf.tensordot(tf.cast(pred_masks, tf.float32), brats_class_mapping[brats_class], axes=1)

    return tf.tuple(tensors=[tf.cast(bool_true_masks, tf.int32), tf.cast(bool_pred_masks, tf.int32)])


@tf.function
def dice_coef(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates Dice score from ground truth and predictions tensors.
    :param true: Ground truth tensor
    :param pred: Prediction tensor
    :return: Dice score
    """
    tp = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(tf.equal(true, 1), tf.equal(pred, 1)), tf.float32),
                       axis=(-1, -2))
    fp = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(tf.equal(true, 0), tf.equal(pred, 1)), tf.float32),
                       axis=(-1, -2))
    fn = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(tf.equal(true, 1), tf.equal(pred, 0)), tf.float32),
                       axis=(-1, -2))

    dice_coefs = tf.math.divide_no_nan(2 * tp, 2 * tp + fp + fn) + tf.cast(tf.equal(tp + fp + fn, 0), tf.float32)

    return dice_coefs


def get_model_fn() -> Callable:
    """
    Get model function.
    :return: Model function
    """
    def model_fn(features: dict, mode: str, params: dict) -> tf.estimator.EstimatorSpec:
        """
        Model function for the Tenorflow estimator.
        :param features: Dictionary with 2D mri-slices and ground truth segmentation maps.
        :param mode: Training or evaluation
        :param params: Includes model dir and number training steps per epoch
        :return: A Tensorflow estimator specification
        """
        model = DenseTiramisu(growth_k=16, layers_per_block=[4, 5, 7, 10, 12, 15], num_classes=FLAGS.num_classes)

        global_step = tf.compat.v1.train.get_global_step()
        metric_dict = {}
        training_summaries = []
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if FLAGS.unsup_ratio > 0 and is_training:
            all_images = tf.concat([features["image"],
                                    features["ori_image"],
                                    features["aug_image"]], 0)
        else:
            all_images = features["image"]

        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            all_logits = model.model(all_images, is_training)

        sup_bsz = tf.shape(input=features["image"])[0]
        sup_logits = all_logits[:sup_bsz]

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = tf.argmax(input=sup_logits, axis=-1, output_type=tf.int32)
            output = {
                'prediction': predictions,
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=output,
                export_outputs={
                    'output': tf.estimator.export.PredictOutput(output)
                })

        # Supervised loss
        sup_maps = features['seg_map']
        sup_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sup_maps,
            logits=sup_logits)
        sup_prob = tf.nn.softmax(sup_logits, axis=-1)
        training_summaries.append(
            tf.compat.v1.summary.scalar('sup/pred_prob', tf.reduce_mean(input_tensor=tf.reduce_mean(
                input_tensor=tf.reduce_max(input_tensor=sup_prob, axis=-1), axis=(-1, -2)))))
        if FLAGS.tsa and is_training:
            # TODO: Implement TSA
            sup_loss, avg_sup_loss = anneal_sup_loss(sup_logits, sup_maps, sup_loss,
                                                     global_step, training_summaries)
        else:
            avg_sup_loss = tf.reduce_mean(input_tensor=tf.reduce_mean(input_tensor=sup_loss, axis=(-1, -2)))
        total_loss = avg_sup_loss
        training_summaries.append(tf.compat.v1.summary.scalar('sup/loss', avg_sup_loss))

        # Unsupervised loss
        if FLAGS.unsup_ratio > 0 and is_training:
            aug_bsz = tf.shape(input=features["ori_image"])[0]

            ori_logits = all_logits[sup_bsz: sup_bsz + aug_bsz]
            aug_logits = all_logits[sup_bsz + aug_bsz:]
            if FLAGS.softmax_temp != -1:
                # TODO: Find out if this is needed
                ori_logits_tgt = ori_logits / FLAGS.softmax_temp
            else:
                ori_logits_tgt = ori_logits
            ori_logits_aug = tf.compat.v1.py_func(unsup_logits_aug, [ori_logits_tgt, features['seed_sq_ent']],
                                                  tf.float32,
                                                  stateful=False)
            ori_logits_aug.set_shape(ori_logits_tgt.shape)
            ori_prob = tf.nn.softmax(ori_logits_aug, axis=-1)
            aug_prob = tf.nn.softmax(aug_logits, axis=-1)

            training_summaries.append(
                tf.compat.v1.summary.scalar('unsup/ori_prob', tf.reduce_mean(input_tensor=tf.reduce_mean(
                    input_tensor=tf.reduce_max(input_tensor=ori_prob, axis=-1), axis=(-1, -2)))))
            training_summaries.append(
                tf.compat.v1.summary.scalar('unsup/aug_prob', tf.reduce_mean(input_tensor=tf.reduce_mean(
                    input_tensor=tf.reduce_max(input_tensor=aug_prob, axis=-1), axis=(-1, -2)))))

            aug_loss = _kl_divergence_with_logits(
                p_logits=tf.stop_gradient(ori_logits_aug),
                q_logits=aug_logits)

            if FLAGS.unsup_crop:
                aug_image_sum = tf.reduce_sum(input_tensor=features['aug_image'], axis=-1)
                loss_mask = tf.cast(tf.not_equal(aug_image_sum, tf.zeros(aug_image_sum.shape)), tf.float32)
                loss_mask = tf.stop_gradient(loss_mask)

                aug_loss = aug_loss * loss_mask
                avg_unsup_loss = tf.reduce_mean(input_tensor=(tf.reduce_sum(input_tensor=aug_loss, axis=(-1, -2)) /
                                                              tf.maximum(
                                                                  tf.reduce_sum(input_tensor=loss_mask, axis=(-1, -2)),
                                                                  1)))

                training_summaries.append(
                    tf.compat.v1.summary.image('unsup/loss_mask', tf.expand_dims(loss_mask, -1), 1))
            else:
                avg_unsup_loss = tf.reduce_mean(input_tensor=tf.reduce_mean(input_tensor=aug_loss, axis=(-1, -2)))

            total_loss += FLAGS.unsup_coeff * avg_unsup_loss
            training_summaries.append(tf.compat.v1.summary.scalar('unsup/loss', avg_unsup_loss))

        # Weight decay
        if FLAGS.do_weight_decay:
            total_loss = utils.decay_weights(
               total_loss,
               FLAGS.weight_decay_rate)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        tf.compat.v1.logging.info("#params: {}".format(num_params))

        if FLAGS.verbose:
            format_str = "{{:<{0}s}}\t{{}}".format(
                max([len(v.name) for v in tf.compat.v1.trainable_variables()]))
            for v in tf.compat.v1.trainable_variables():
                tf.compat.v1.logging.info(format_str.format(v.name, v.get_shape()))

        # Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = tf.argmax(input=sup_logits, axis=-1, output_type=tf.int32)

            loss = tf.compat.v1.metrics.mean(tf.reduce_mean(input_tensor=sup_loss, axis=(-1, -2)))

            brats_classes = ['whole', 'core', 'enhancing']
            dice_scores = []
            mean_dice_scores = {}

            for brats_class in brats_classes:
                true, pred = class_convert(sup_maps, predictions, brats_class)
                dice_score = dice_coef(true, pred)
                dice_scores.append(dice_score)
                mean_dice_scores[brats_class] = tf.compat.v1.metrics.mean(dice_score)

            dice_collective = tf.add_n(dice_scores)
            dice_collective = tf.compat.v1.metrics.mean(tf.divide(dice_collective, 3))

            eval_metrics = {
                "eval/classify_loss": loss,
                "eval/classify_whole_dice": mean_dice_scores['whole'],
                "eval/classify_core_dice": mean_dice_scores['core'],
                "eval/classify_enhancing_dice": mean_dice_scores['enhancing'],
                "eval/classify_collective_dice": dice_collective
            }

            if FLAGS.plot_eval_images:
                tf.compat.v1.summary.image('eval/input', tf.expand_dims(all_images[..., 0], -1), 2)
                tf.compat.v1.summary.image('eval/gt_mask', tf.cast(tf.expand_dims(sup_maps, -1), tf.float32), 2)
                tf.compat.v1.summary.image('eval/pred_mask', tf.cast(tf.expand_dims(predictions, -1), tf.float32), 2)

            eval_summary_hook = tf.estimator.SummarySaverHook(
                save_secs=120,
                output_dir=FLAGS.model_dir,
                summary_op=tf.compat.v1.summary.merge_all())

            # Constructing evaluation EstimatorSpec.
            eval_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=avg_sup_loss,
                eval_metric_ops=eval_metrics,
                evaluation_hooks=[eval_summary_hook])

            return eval_spec

        # Learning rate
        learning_rate = tf.Variable(FLAGS.learning_rate)
        if FLAGS.exp_lr_decay:
            learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, global_step,
                                                                 params['epoch_steps'], 0.95)

        training_summaries.append(tf.compat.v1.summary.scalar('lr/learning_rate', learning_rate))

        # Backprop
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        gradients, variables = zip(*grads_and_vars)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(
                zip(gradients, variables), global_step=tf.compat.v1.train.get_global_step())

        # Creating training logging hook
        sup_pred = tf.argmax(input=sup_logits, axis=-1, output_type=sup_maps.dtype)
        is_correct = tf.cast(tf.equal(sup_pred, sup_maps), dtype=tf.float32)
        acc = tf.reduce_mean(input_tensor=is_correct)
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
        logging_hook = tf.estimator.LoggingTensorHook(
            tensors=metric_dict,
            every_n_iter=FLAGS.train_summary_steps,
            formatter=formatter)

        if FLAGS.unsup_ratio > 0 and FLAGS.plot_train_images:
            training_summaries.append(
                tf.compat.v1.summary.image('unsup/ori_image', tf.expand_dims(features['ori_image'][..., 0], -1), 1))
            training_summaries.append(
                tf.compat.v1.summary.image('unsup/aug_image', tf.expand_dims(features['aug_image'][..., 0], -1), 1))
            training_summaries.append(tf.compat.v1.summary.image('unsup/ori_mask', tf.cast(
                tf.expand_dims(tf.argmax(input=ori_logits_aug, axis=-1, output_type=tf.int32), -1),
                tf.float32), 1))
            training_summaries.append(tf.compat.v1.summary.image('unsup/aug_mask', tf.cast(
                tf.expand_dims(tf.argmax(input=aug_logits, axis=-1, output_type=tf.int32), -1),
                tf.float32), 1))

            training_summary_hook = tf.estimator.SummarySaverHook(
                save_steps=FLAGS.train_summary_steps,
                output_dir=FLAGS.model_dir,
                summary_op=training_summaries
            )
            training_hooks = [logging_hook, training_summary_hook]
        else:
            training_hooks = [logging_hook]

        # Constucting training EstimatorSpec.
        train_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op,
            training_hooks=training_hooks)

        return train_spec

    return model_fn


def train():
    """
    Training and evaluation.
    """
    with tf.io.gfile.GFile(os.path.join(FLAGS.data_dir, 'data_info.json'), 'r') as fp:
        data_info = json.load(fp)
    if FLAGS.unsup_ratio == 0:
        FLAGS.unsup_cut = 0.0

    # Create input functions
    train_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split="train",
        data_size=data_info['train']['size'],
        batch_size=FLAGS.train_batch_size,
        sup_cut=FLAGS.sup_cut,
        unsup_cut=FLAGS.unsup_cut,
        unsup_ratio=FLAGS.unsup_ratio,
        shuffle_seed=FLAGS.shuffle_seed
    )

    eval_input_fn = data.get_input_fn(
        data_dir=FLAGS.data_dir,
        split="val",
        data_size=data_info['val']['size'],
        batch_size=FLAGS.eval_batch_size,
        sup_cut=1.0,
        unsup_cut=0.0,
        unsup_ratio=0
    )

    eval_size = data_info['val']['size']
    eval_steps = eval_size // FLAGS.eval_batch_size

    epoch_steps = int((data_info['train']['size'] * FLAGS.sup_cut) / FLAGS.train_batch_size)

    # Get model function
    model_fn = get_model_fn()
    estimator = utils.get_estimator(FLAGS, model_fn, epoch_steps)

    # Training
    if FLAGS.do_eval_along_training:

        tf.compat.v1.logging.info("***** Running training & evaluation *****")
        tf.compat.v1.logging.info("  Supervised batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Unsupervised batch size = %d",
                                  FLAGS.train_batch_size * FLAGS.unsup_ratio)
        tf.compat.v1.logging.info("  Num train steps = %d", FLAGS.train_steps)

        # 7 is arbitrary and will be eliminated
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            {'image': tf.zeros([7] + [224, 224, 4], tf.float32)})

        best_exporter = tf.estimator.BestExporter(
            name="best_exporter",
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=1)

        best_ckpt = best_ckpt_copier.BestCheckpointCopier(
            name="best_checkpoint",
            checkpoints_to_keep=1,
            score_metric='loss')

        latest_exporter = tf.estimator.LatestExporter(
            name="latest_exporter",
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=None)

        hooks = []

        if FLAGS.early_stop_steps != -1:
            # Hook to stop training if loss does not decrease in over FLAGS.early_stop_steps steps.
            hooks.append(tf.estimator.experimental.stop_if_no_decrease_hook(estimator, "loss",
                                                                            FLAGS.early_stop_steps,
                                                                            min_steps=FLAGS.min_step))

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_steps, hooks=hooks)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps,
                                          exporters=[best_exporter, latest_exporter, best_ckpt], start_delay_secs=0,
                                          throttle_secs=10)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_predict:
        tf.compat.v1.logging.info('***** Running prediction *****')

        with open('data/processed_data/aug_seed.json', 'r') as fp:
            aug_seeds_dict = json.load(fp)
        aug_seeds_list = list(itertools.chain.from_iterable(list(aug_seeds_dict[FLAGS.pred_dataset].values())))

        if FLAGS.pred_ckpt == 'best':
            out_path = os.path.join(FLAGS.model_dir, 'best_{}_prediction'.format(FLAGS.pred_dataset))
            export_dir = sorted(glob.glob(os.path.join(FLAGS.model_dir, 'export/best_exporter/*')))[-1]
        else:
            out_path = os.path.join(FLAGS.model_dir, '{}_{}_prediction'.format(FLAGS.pred_ckpt, FLAGS.pred_dataset))
            export_dir = os.path.join(FLAGS.model_dir, 'export/latest_exporter/{}'.format(FLAGS.pred_ckpt))

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        model = tf.compat.v2.saved_model.load(export_dir)
        predict = model.signatures["serving_default"]

        tf.compat.v1.logging.info('Predicting images')

        test_input_fn = data.get_input_fn(
            data_dir=FLAGS.data_dir,
            split=FLAGS.pred_dataset,
            data_size=data_info[FLAGS.pred_dataset]['size'],
            batch_size=FLAGS.pred_batch_size,
            sup_cut=1.0,
            unsup_cut=0.0,
            unsup_ratio=0
        )

        dataset = test_input_fn()
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

        preds = []
        aug_preds = []

        example_cnt = 1
        patient_cnt = 0
        img_cnt = 0

        def pad_and_export(patient_pred: np.ndarray, mode: str):
            """
            Pad and export predicted segmentation maps as .nii.gz-files.
            :param patient_pred: Predicted segmentation maps
            :param mode: Standard prediction, augmented prediction
             or predicted augmentation
            """
            # Make the prediction dims (155,240,240) again for BRATS evaluation
            patient_pred = np.pad(patient_pred, ((0, 0), (8, 8), (8, 8)))

            below_padding = np.zeros(
                (data_info[FLAGS.pred_dataset]['crop_idx'][patient_cnt][0], 240, 240))
            above_padding = np.zeros(
                (155 - data_info[FLAGS.pred_dataset]['crop_idx'][patient_cnt][1], 240, 240))
            patient_pred = np.concatenate([below_padding, patient_pred, above_padding])

            if not os.path.exists(os.path.join(out_path, mode)):
                os.makedirs(os.path.join(out_path, mode))

            patient_pred_nii = sitk.GetImageFromArray(patient_pred)
            sitk.WriteImage(patient_pred_nii, os.path.join(out_path, mode, '{}.nii.gz'.format(patient_id)))

            tf.compat.v1.logging.info('Exported patient {}'.format(patient_id))

        for sample in iterator:
            imgs = sample['image'].numpy()

            aug_seeds = aug_seeds_list[img_cnt:img_cnt + imgs.shape[0]]
            imgs_aug = data_aug(imgs, aug_seeds, is_seg_maps=False)

            pred_batch = predict(image=tf.constant(imgs))['prediction'].numpy()
            aug_pred_batch = predict(image=tf.constant(imgs_aug))['prediction'].numpy()

            for i in range(pred_batch.shape[0]):
                pred = pred_batch[i, ...]
                aug_pred = aug_pred_batch[i, ...]

                pred[np.where(pred == 3)] = 4
                aug_pred[np.where(aug_pred == 3)] = 4

                preds.append(pred)
                aug_preds.append(aug_pred)

                if example_cnt == data_info[FLAGS.pred_dataset]['slices'][patient_cnt]:
                    patient_id = data_info[FLAGS.pred_dataset]['paths'][patient_cnt].split('/')[-1]

                    patient_pred = np.stack(preds)
                    patient_aug_pred = np.stack(aug_preds)
                    patient_pred_aug = data_aug(patient_pred, aug_seeds_dict[FLAGS.pred_dataset][patient_id],
                                                is_seg_maps=True)

                    pad_and_export(patient_pred, 'standard')
                    pad_and_export(patient_aug_pred, 'aug_pred')
                    pad_and_export(patient_pred_aug, 'pred_aug')

                    example_cnt = 1
                    patient_cnt += 1
                    preds = []
                    aug_preds = []
                else:
                    example_cnt += 1
            img_cnt += imgs.shape[0]

        if FLAGS.pred_dataset == 'val':
            tf.compat.v1.logging.info('Calculating standard Dice scores')
            calc_and_export_standard_dice(os.path.join(out_path, 'standard'))

        tf.compat.v1.logging.info('Calculating equivariance Dice scores')
        calc_and_export_equivariance_dice(os.path.join(out_path, 'pred_aug'), os.path.join(out_path, 'aug_pred'))


def main(_):
    """
    Main function.
    """
    if FLAGS.do_eval_along_training:
        tf.io.gfile.makedirs(FLAGS.model_dir)
        flags_txt = FLAGS.flags_into_string()
        with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, "FLAGS.txt"), "w") as ouf:
            ouf.write(flags_txt)

    train()


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()
