from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops


class CombinedDenseSameInput(base.Layer):
    """Combination layer that combines tf.Dense operators for two different graphs Densely-connected layer class.

    This layer implements the operation:
    `outputs = activation(inputs * kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).

    Arguments:
    layers_to_combine
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    """

    def __init__(self, layers_to_combine,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(CombinedDenseSameInput, self).__init__(trainable=trainable, name=name, **kwargs)
        self.dense_layers = layers_to_combine
        self.dense_kernels = [d.add_variable('kernel', [self.dense_layers[0].input_spec.axes[-1], d.units])
                              for d in self.dense_layers]

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `CombinedDenseSameInput` should be defined. '
                             'Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.built = True

    def call(self, inputs, **kwargs):
        combined_kernel = tf.concat(axis=1, values=self.dense_kernels, name='combined_kernel')
        input_shape = tensor_shape.TensorShape(inputs.shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, combined_kernel, [[len(shape) - 1],
                                                             [0]])
            # # Reshape the output back to the original ndim of the input.
            # if context.in_graph_mode():
            #   output_shape = shape[:-1] + [self.units]
            #   outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, combined_kernel)
        prev = None
        output_ops = []
        bias_combined = False
        activations_combined = False
        if all([d.use_bias for d in self.dense_layers]):
            combined_bias = tf.concat(axis=0, values=[d.bias for d in self.dense_layers])
            outputs = nn.bias_add(outputs, combined_bias)
            bias_combined = True
        activations = {d.activation for d in self.dense_layers}
        if None not in activations and len(activations) == 1:
            outputs = activations.pop()(outputs)
            activations_combined = True
        for d in self.dense_layers:
            if prev is None:
                layer_output = outputs[:, :d.units]
            else:
                layer_output = outputs[:, prev:(prev+d.units)]
            prev = d.units
            if d.use_bias and not bias_combined:
                print(d.bias)
                print(layer_output)
                layer_output = nn.bias_add(layer_output, d.bias)
            if d.activation is not None and not activations_combined:
                layer_output = d.activation(layer_output)
            output_ops.append(layer_output)
        # this can be significantly more optimized - we could bias_add all at once
        return output_ops

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)


def combinedDenseSameInput(inputs, layers_to_combine, trainable=True, name=None, reuse=None):
    """Functional interface for the combined densely-connected layer, using the same input.

    This layer implements the operation:
    `outputs = activation(inputs.kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).

    Arguments:
    inputs: Tensor input.
    d1: Dense layer to merge
    d2: Dense layer to merge
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

    Returns:
    Output tensor the same shape as `inputs` except the last dimension is of
    size `units`.

    Raises:
    ValueError: if eager execution is enabled.
    """
    layer = CombinedDenseSameInput(layers_to_combine,
                                   trainable=trainable,
                                   name=name,
                                   dtype=inputs.dtype.base_dtype,
                                   _scope=name,
                                   _reuse=reuse)
    return layer.apply(inputs)