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


def plateau_decay(learning_rate, global_step, loss, factor=0.5, patience=6000, min_delta=0,
                  cooldown=0, min_lr=0):
    if not isinstance(learning_rate, tf.Tensor):
        learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(learning_rate), trainable=False,
                                        collections=[tf.GraphKeys.LOCAL_VARIABLES])

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
                                global_step + patience, 'Loss:', loss)
            ]):
                return tf.identity(learning_rate)

        def _decay():
            with tf.control_dependencies([
                tf.assign(best, loss),
                tf.assign(learning_rate, tf.maximum(tf.multiply(learning_rate, factor), min_lr)),
                tf.assign(step, global_step + cooldown),
                tf.print('Plateau Decay: Decayed LR - Step:', global_step, 'Next Decay Step:',
                                global_step + cooldown + patience, 'Learning Rate:', learning_rate)
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
