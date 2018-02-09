import tensorflow as tf
import numpy as np
import argparse

from eval_models import eval_model, eval_sequential
from model import Model


class SimpleNN(Model):
    def __init__(self, input_shape, hl_sizes, activation=None):
        super(SimpleNN, self).__init__()
        self.has_graph = False
        self.input_shape = input_shape
        self.hl_sizes = hl_sizes
        self.activation = activation
        self.input_dtype = np.float32
        self.tf_dtype = tf.float32

    def create_graph(self):
        flattened_inp_size = reduce(mul, self.input_shape)
        shape = [None, flattened_inp_size]
        self.input_tensor = tf.placeholder(self.tf_dtype, shape)

        # Create hidden layers
        curr_tensor = self.input_tensor
        for hl_size in self.hl_sizes:
            # Create a weights matrix for the hidden layer
            hl_weights = tf.Variable(tf.zeros(flattened_inp_size, hl_size))
            curr_tensor = tf.matmul(curr_tensor, hl_weights)
            curr_tensor = activation(curr_tensor)

        last_hl_size = self.hl_sizes[-1]
        output_weights = tf.Variable(last_hl_size, 1)
        self.output_tensor = tf.matmul(curr_tensor, output_weights)

        self.has_graph = True

    def get_input_tensor(self):
        if not self.has_graph:
            raise Exception(
                "Attempted to get input tensor from model whose graph has not been created"
            )

        return self.input_tensor

    def get_output_tensor(self):
        if not self.has_graph:
            raise Exception(
                "Attempted to get output tensor from model whose graph has not been created"
            )

        return self.output_tensor

    def get_inputs(self, num_inputs):
        inps_list = [
            np.random.rand(*inputs_shape).flatten() for _ in range(num_inputs)
        ]
        return np.array(inps_list, dtype=self.input_dtype)


def main():
    parser = argparse.ArgumentParser(
        description=
        'Test merged inference vs sequential inference on two small neural networks'
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="The query batch size to evaluate")
    parser.add_argument(
        "-g",
        "--gpu_num",
        type=int,
        default=0,
        help="The number of the GPU on which to execute the experiment")

    args = parser.parse()

    # Use the same input shape for both nets
    input_shape = (128, 128, 3)
    batch_size = 8

    nn_a = SimpleNN(
        input_shape=input_shape, hl_sizes=[20, 10], activation=None)
    nn_b = SimpleNN(input_shape=input_shape, hl_sizes=[8, 6], activation=None)

    eval_sequential(
        [nn_a, nn_b], batch_size=args.batch_size, gpu_num=args.gpu_num)

if __name__ == "__main__":
    main()
