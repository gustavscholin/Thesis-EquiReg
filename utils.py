"""Helper functions."""
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def save_plt_as_img(path, name, img=None, seg=None):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if img is not None:
        ax.imshow(img, 'gray', interpolation='none')
    if seg is not None:
        ax.imshow(seg, 'jet', vmin=0, vmax=3, interpolation='none', alpha=0.5)
    plt.savefig(os.path.join(path, '{}.jpg'.format(name)), bbox_inches='tight')
    plt.close()


def collective_dice_smaller(best_eval_result, current_eval_result):
  """Compares two evaluation results and returns true if the 2nd one is smaller.

  Both evaluation results should have the values for MetricKeys.LOSS, which are
  used for comparison.

  Args:
    best_eval_result: best eval metrics.
    current_eval_result: current eval metrics.

  Returns:
    True if the loss of current_eval_result is smaller; otherwise, False.

  Raises:
    ValueError: If input eval result is None or no loss is available.
  """
  default_key = 'eval/classify_collective_dice'
  if not best_eval_result or default_key not in best_eval_result:
    raise ValueError(
        'best_eval_result cannot be empty or no loss is found in it.')

  if not current_eval_result or default_key not in current_eval_result:
    raise ValueError(
        'current_eval_result cannot be empty or no loss is found in it.')

  return best_eval_result[default_key] > current_eval_result[default_key]


def decay_weights(cost, weight_decay_rate):
    """Calculates the loss for l2 weight decay and adds it to `cost`."""
    costs = []
    for var in tf.compat.v1.trainable_variables():
        costs.append(tf.nn.l2_loss(var))
    cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
    return cost


def get_estimator(FLAGS, model_fn, epoch_steps, model_dir=None):
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
        params={"model_dir": model_dir or FLAGS.model_dir,
                'epoch_steps': epoch_steps})
    return estimator
