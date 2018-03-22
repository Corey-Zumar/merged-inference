import tensorflow as tf
import numpy as np

# Code that attempts to find candidate layers to merge together for inference time
# worthy experiment: How does cuBLAS matrix multiply performance scale with matrix size?
# Hard constraints: GPU multiply unit size (12GB), multiplication precision
# Soft constraints: For small models, we could probably run quite a few kernels

# TODO: Remove layers with unused output nodes, 

# vertical layer fusion is probably just making a bunch of ops work in a single CUDA program

# horizontal layer fusion most optimal when you take the same source tensor

# get colocated models

# iterate through model graphs and categorize them

# iterate through categorized layers and evaluate them for combination

# 

def fuse_distinct_matmul(graph, op1_name, op2_name):
	"""
	op1 : str
		The name of the first matrix multiplication operation
	op2 : str
		The name of the second matrix multiplication operation
	"""

	with tf.Session() as sess:
		tf.import_graph_def(graph, name="")

		print(tf.get_default_graph().get_operations())

		op1 = tf.get_default_graph().get_operation_by_name(op1_name)
		op2 = tf.get_default_graph().get_operation_by_name(op2_name)

		left1, right1 = op1.inputs
		left2, right2 = op2.inputs

		sess.run(tf.global_variables_initializer())

		weights1 = sess.run(left1)
		weights2 = sess.run(left2)

		x1, y1 = weights1.shape
		x2, y2 = weights2.shape

		s = min(y1, y2)
		column_diff = abs(y2 - y1)
		inputs_padding = tf.constant([[0, column_diff], [0, 0]])

		if column_diff > 0:
			if y1 == s:
				weights_padding = np.zeros((x1, column_diff))
				weights1 = np.concatenate((weights1, weights_padding), axis=1)
				right1 = tf.pad(right1, inputs_padding)
			elif y2 == s:
				weights_padding = np.zeros((x2, column_diff))
				weights2 = np.concatenate((weights2, weights_padding), axis=1)
				right2 = tf.pad(right2, inputs_padding)

		merged_weights = np.array(np.concatenate((weights1, weights2), axis=0), dtype=np.float32)
		merged_tensor = tf.Variable(merged_weights)

		print(merged_tensor.get_shape())

		merged_matmul = tf.matmul(merged_tensor, tf.concat([right1, right2], axis=1))

		right1_shape = tf.shape(right1)
		right2_shape = tf.shape(right2)

		new_op1, new_op2 = tf.split(merged_matmul, [x1, x2], axis=0)
		new_op1, _ = tf.split(new_op1, [right1_shape[0], right2_shape[0]], axis=1)
		_, new_op2 = tf.split(new_op2, [right1_shape[0], right2_shape[0]], axis=1)

		print(op1.outputs)


if __name__ == "__main__":
	with tf.Session() as sess:
		with tf.variable_scope("TEST"):
			w_vals = np.zeros((12, 10), dtype=np.float32)
			w_vals[0][0] = 8.2
			w = tf.Variable(w_vals)
			x = tf.Variable(tf.zeros([4, 11]))
			y = tf.zeros([10, 9])
			b = tf.zeros([11, 8])
			z = tf.matmul(x,b)
			a = tf.matmul(w,y)

			print(z.shape)
			print(a.shape)

			sess.run(tf.global_variables_initializer())

			frozen_graph = tf.graph_util.convert_variables_to_constants(
				sess, tf.get_default_graph().as_graph_def(), ["TEST/MatMul", "TEST/MatMul_1"])

			#print(frozen_graph.node)

			fuse_distinct_matmul(frozen_graph, "TEST/MatMul", "TEST/MatMul_1")


