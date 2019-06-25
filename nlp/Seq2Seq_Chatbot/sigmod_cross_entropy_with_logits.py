# -*- coding:utf-8 -*-
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()
logits = tfe.Variable(np.random.rand(1, 3), dtype=tf.float32)

labels = [[1.0, 0.0, 0.0]]

_sentinel = None
# tf.nn.sigmoid_cross_entropy_with_logits()
# nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
#                          labels, logits)
# pylint: enable=protected-access

with ops.name_scope("logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
        labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                         (logits.get_shape(), labels.get_shape()))

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    loss = math_ops.add(
        relu_logits - logits * labels,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=name)
    print(loss)
