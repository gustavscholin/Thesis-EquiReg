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
"""Helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import collections


def _summaries(eval_dir):
    """Yields `tensorflow.Event` protos from event files in the eval dir.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Yields:
    `tensorflow.Event` object read from the event files.
  """
    if tf.gfile.Exists(eval_dir):
        for event_file in tf.gfile.Glob(
                os.path.join(eval_dir, 'events.out.tfevents.*')):
            for event in tf.train.summary_iterator(event_file):
                yield event


def read_eval_metrics(eval_dir):
    """Helper to read eval metrics from eval summary files.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Returns:
    A `dict` with global steps mapping to `dict` of metric names and values.
  """
    eval_metrics_dict = collections.defaultdict(dict)
    for event in _summaries(eval_dir):
        if not event.HasField('summary'):
            continue
        metrics = {}
        for value in event.summary.value:
            if value.HasField('simple_value'):
                metrics[value.tag] = value.simple_value
        if metrics:
            eval_metrics_dict[event.step].update(metrics)
    return collections.OrderedDict(
        sorted(eval_metrics_dict.items(), key=lambda t: t[0]))


def plateau_decay(learning_rate, global_step, eval_dir, factor=0.5, patience=6000, min_delta=0,
                  cooldown=0, min_lr=0):

    if not isinstance(learning_rate, tf.Tensor):
        learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(learning_rate), trainable=False,
                                        collections=[tf.GraphKeys.LOCAL_VARIABLES])

    eval_results = read_eval_metrics(eval_dir)
    if eval_results:
        loss = tf.constant(eval_results[next(reversed(eval_results))]['loss'])
    else:
        return tf.identity(learning_rate)

    with tf.variable_scope('plateau_decay'):
        step = tf.get_variable('step', trainable=False, initializer=global_step,
                               collections=[tf.GraphKeys.LOCAL_VARIABLES])
        best = tf.get_variable('best', trainable=False, initializer=tf.constant(np.Inf, tf.float32),
                               collections=[tf.GraphKeys.LOCAL_VARIABLES])

        def _update_best():
            with tf.control_dependencies([
                tf.assign(best, loss),
                tf.assign(step, global_step),
                tf.print('Plateau Decay: Updated Best - Step:', global_step, 'Next Decay Step:',
                         global_step + patience, 'Loss:', loss, output_stream='file:///home/justin/gustav_workspace/thesis-uda-segmentation/decay.out')
            ]):
                return tf.identity(learning_rate)

        def _decay():
            with tf.control_dependencies([
                tf.assign(best, loss),
                tf.assign(learning_rate, tf.maximum(tf.multiply(learning_rate, factor), min_lr)),
                tf.assign(step, global_step + cooldown),
                tf.print('Plateau Decay: Decayed LR - Step:', global_step, 'Next Decay Step:',
                         global_step + cooldown + patience, 'Learning Rate:', learning_rate, output_stream='file:///home/justin/gustav_workspace/thesis-uda-segmentation/decay.out')
            ]):
                return tf.identity(learning_rate)

        def _no_op(): return tf.identity(learning_rate)

        met_threshold = tf.less(loss, best - min_delta)
        should_decay = tf.greater_equal(global_step - step, patience)

        return tf.cond(met_threshold, _update_best, lambda: tf.cond(should_decay, _decay, _no_op))


def decay_weights(cost, weight_decay_rate):
    """Calculates the loss for l2 weight decay and adds it to `cost`."""
    costs = []
    for var in tf.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
    cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
    return cost


def get_estimator(FLAGS, model_fn, model_dir=None):
    ##### Create Estimator
    # Estimator Configuration
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir or FLAGS.model_dir,
        keep_checkpoint_max=FLAGS.max_save,
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.save_steps
    )

    # Estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"model_dir": model_dir or FLAGS.model_dir})
    return estimator
