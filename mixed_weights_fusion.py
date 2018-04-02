import tensorflow as tf
import numpy as np
import itertools

from datetime import datetime

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

		merged_matmul = tf.matmul(merged_tensor, tf.concat([right1, right2], axis=1))

		right1_shape = tf.shape(right1)
		right2_shape = tf.shape(right2)

		new_op1, new_op2 = tf.split(merged_matmul, [x1, x2], axis=0)

		new_op1, _ = tf.split(new_op1, [right1_shape[1], right2_shape[1]], axis=1)
		_, new_op2 = tf.split(new_op2, [right1_shape[1], right2_shape[1]], axis=1)

		return new_op1, new_op2


def evaluate_matmul():
	sizes = [(256 * 256, 500), (128 * 128, 200), (64 * 64, 100)]

	all_sizes = itertools.product(sizes, sizes)

	idx = 0
	for s1, s2 in all_sizes:
		if idx == 0:
			idx += 1
			continue
		m1, n1 = s1
		m2, n2 = s2
		with tf.Session() as sess:
			with tf.variable_scope("TEST"):
				w1 = tf.Variable(np.array(np.random.rand(m1, n1), dtype=np.float32))
				w2 = tf.Variable(np.array(np.random.rand(m2, n2), dtype=np.float32))
				
				inp1 = tf.placeholder(tf.float32, [n1, None])
				inp2 = tf.placeholder(tf.float32, [n2, None])

				result1 = tf.matmul(w1, inp1)
				result2 = tf.matmul(w2, inp2)
				
				sess.run(tf.global_variables_initializer())

				frozen_graph = tf.graph_util.convert_variables_to_constants(
					sess, tf.get_default_graph().as_graph_def(), ["TEST/MatMul", "TEST/MatMul_1"])

				new_op1, new_op2 = fuse_distinct_matmul(frozen_graph, "TEST/MatMul", "TEST/MatMul_1")

				sess.run(tf.global_variables_initializer())
		
			feed_dict = {
					inp1 : np.random.rand(n1,1),
					inp2 : np.random.rand(n2,1)
				    }
		
			before = datetime.now()

			out1, out2 = sess.run([new_op1, new_op2], feed_dict=feed_dict)

			mid = datetime.now()

			out1, out2 = sess.run([result1, result2], feed_dict=feed_dict)

			after = datetime.now()

			merged_lat = (mid - before).total_seconds()
			iso_lat = (after - mid).total_seconds()

			print(merged_lat, iso_lat)
			break

if __name__ == "__main__":
	evaluate_matmul()
	# with tf.Session() as sess:
	# 	with tf.variable_scope("TEST"):
	# 		w_vals = np.zeros((12, 10), dtype=np.float32)
	# 		w_vals[0][0] = 8.2
	# 		w = tf.Variable(w_vals)
	# 		x = tf.Variable(tf.zeros([4, 11]))
	# 		y = tf.zeros([10, 9])
	# 		b = tf.zeros([11, 8])
	# 		z = tf.matmul(x,b)
	# 		a = tf.matmul(w,y)
        #
	# 		print(z.shape)
	# 		print(a.shape)
        #
	# 		sess.run(tf.global_variables_initializer())
        #
	# 		frozen_graph = tf.graph_util.convert_variables_to_constants(
	# 			sess, tf.get_default_graph().as_graph_def(), ["TEST/MatMul", "TEST/MatMul_1"])
        #
	# 		#print(frozen_graph.node)
        #
	# 		fuse_distinct_matmul(frozen_graph, "TEST/MatMul", "TEST/MatMul_1")


