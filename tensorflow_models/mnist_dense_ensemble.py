from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import argparse
import os
from merged_dense import combinedDenseSameInput

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SIZE = 28
NUM_LABELS = 10
FLAGS = None


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def dense_model_fn(placeholders):
    x = placeholders['x']
    y_ = placeholders['labels']
    prob = placeholders['dropout']

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [FLAGS.batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [FLAGS.batch_size, 1024]
    dense_layers_1 = []
    for i in range(FLAGS.n_ensemble):
        with tf.variable_scope('dense1_' + str(i)) as scope:
            if FLAGS.combine_dense:
                d = tf.layers.Dense(units=1024, activation=tf.nn.relu, name=scope.name)
                d.apply(x)
                dense_layers_1.append(d)
            else:
                dense_layers_1.append(tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, name=scope.name))
    if FLAGS.combine_dense:
        dense_layers_1 = combinedDenseSameInput(inputs=x, layers_to_combine=dense_layers_1)
    dense_layers_2 = []
    for i in range(FLAGS.n_ensemble):
        with tf.variable_scope('dense2_' + str(i)) as scope:
            dense_layers_2.append(tf.layers.dense(inputs=dense_layers_1[i], units=512, activation=tf.nn.relu, name=scope.name))

    # Add dropout operation; 0.6 probability that element will be kept
    dropouts = []
    for i in range(FLAGS.n_ensemble):
        with tf.variable_scope('dropout_' + str(i)) as scope:
            dropouts.append(tf.layers.dropout(inputs=dense_layers_2[i], rate=prob, name=scope.name))

    # Logits layer
    # Input Tensor Shape: [FLAGS.batch_size, 1024]
    # Output Tensor Shape: [FLAGS.batch_size, 10]

    logits_layers = []
    for i in range(FLAGS.n_ensemble):
        with tf.variable_scope('logits_' + str(i)) as scope:
            logits_layers.append(tf.layers.dense(inputs=dropouts[i], units=10, name=scope.name))

    # averaging step
    k = 0
    while len(logits_layers) >= 2:
        logit_0, logit_1 = logits_layers[0], logits_layers[1]
        with tf.variable_scope('add_' + str(k)) as scope:
            combined_logit = tf.add(logit_0, logit_1, name=scope.name)
        k += 1
        logits_layers.remove(logit_0)
        logits_layers.remove(logit_1)
        logits_layers.append(combined_logit)
    all_added = logits_layers[0]
    combined_logits = tf.divide(all_added, float(FLAGS.n_ensemble), name= 'average_tensor')
    classes = tf.argmax(input=combined_logits, axis=1, name="output_classes")
    softmax_tensor = tf.nn.softmax(combined_logits, name="softmax_tensor")

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.

    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < FLAGS.batch_size:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        preds = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in xrange(0, size, FLAGS.batch_size):
            end = begin + FLAGS.batch_size
            if end <= size:
                batch_data = data[begin:end, ...]
                preds[begin:end, :] = sess.run(softmax_tensor, feed_dict={x: batch_data})
            else:
                batch_data = data[-FLAGS.batch_size:, ...]
                batch_predictions = sess.run(softmax_tensor, feed_dict={x: batch_data})
                preds[begin:, :] = batch_predictions[begin - size:, :]
        return preds
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": classes,
        # Add `softmax_tensor` to the graph.
        "probabilities": softmax_tensor
    }
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=combined_logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=y_, predictions=predictions["classes"])}
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return loss, predictions, train_op, eval_metric_ops, eval_in_batches


def main(_):
    # Load training and eval data
    # determine version
    model_version = 1
    if os.path.exists(FLAGS.serving_model_path):
        for f in os.listdir(FLAGS.serving_model_path):
            max_version = 0
            try:
                if int(f):
                    if max_version < int(f):
                        max_version = int(f)
            except ValueError:
                continue
        model_version = max_version + 1
    else:
        os.makedirs(FLAGS.serving_model_path)
    FLAGS.serving_model_path += '/' + str(model_version) + '/'
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    val_data = mnist.train.images[:FLAGS.validation_size]  # Returns np.array
    val_labels = np.asarray(mnist.train.labels[:FLAGS.validation_size], dtype=np.uint8)
    train_data = mnist.train.images[FLAGS.validation_size:]
    train_labels = np.asarray(mnist.train.labels[FLAGS.validation_size:], dtype=np.uint8)
    train_size = train_labels.shape[0]
    test_data = mnist.test.images  # Returns np.array
    test_labels = np.asarray(mnist.test.labels, dtype=np.uint8)
    print(FLAGS)
    start_time = time.time()
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE * IMAGE_SIZE))
        y_ = tf.placeholder(tf.uint8, shape=(None,))
        prob = tf.placeholder_with_default(1.0, shape=())
        loss, predictions, train_op, eval_metric_ops, eval_in_batches = dense_model_fn({'x' : x, 'labels' : y_, 'dropout' : prob})

        # Initialize variables
        print('Initializing the model')
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run([init_op])
        # Train the model
        for step in xrange(int(FLAGS.epochs * train_size) // FLAGS.batch_size):
            offset = (step * FLAGS.batch_size) % (train_size - FLAGS.batch_size)
            batch_data = train_data[offset:(offset + FLAGS.batch_size)]
            batch_labels = train_labels[offset:(offset + FLAGS.batch_size)]
            feed_dict = {x : batch_data, y_ : batch_labels, prob : 0.5}
            sess.run(train_op, feed_dict)
            if FLAGS.verbose:
                print('Step %d (epoch %.2f)' % (step, float(step) * FLAGS.batch_size / train_size))
            # Run batch evaluations without backprop on the loss
            if step % FLAGS.eval_frequency == 0:
                predict_feed_dict = feed_dict.copy()
                predict_feed_dict[prob] = 1
                l, batch_preds = sess.run([loss, predictions['probabilities']], feed_dict=predict_feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                  (step, float(step) * FLAGS.batch_size / train_size,
                   1000 * elapsed_time / FLAGS.eval_frequency))
                print('Minibatch loss: %.3f' % (l))
                print('Minibatch error: %.1f%%' % error_rate(batch_preds, batch_labels))
                print('Validation error: %.1f%%' % error_rate(eval_in_batches(val_data, sess), val_labels))
            if step % FLAGS.save_frequency == 0:
                print('Saving model.')
                save_path = saver.save(sess, FLAGS.save_path)
        tf.saved_model.simple_save(
            sess,
            FLAGS.serving_model_path,
            inputs={'inputs': x, 'labels': y_},
            outputs={'classes': predictions['classes']})
        # Compute error over the held out test set
        print('Test error: %.1f%%' % error_rate(eval_in_batches(test_data, sess), test_labels))
        # Save final model
        save_path = saver.save(sess, FLAGS.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose',
        default=False,
        action='store_true',
        help='Print step information more often')
    parser.add_argument(
        '--n_ensemble',
        default=4,
        type=int,
        help='Specify the number of ensembles to include')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Specify batch size for evaluation')
    parser.add_argument(
        '--epochs',
        default=4,
        type=int,
        help='Specify number of epochs to train for')
    parser.add_argument(
        '--eval_frequency',
        default=100,
        type=int,
        help='Specify how often to evaluate model performance')
    parser.add_argument(
        '--save_frequency',
        default=1000,
        type=int,
        help='Specify how often to evaluate model performance')
    parser.add_argument(
        '--validation_size',
        default=1000,
        type=int,
        help='Number of samples to set aside for validation')
    parser.add_argument(
        '--save_path',
        default='models/dense_model/',
        type=str,
        help='Name of prefix to save model checkpoints under.')
    parser.add_argument(
        '--serving_model_path',
        default='models/model_serving/',
        type=str,
        help='Name of prefix to save model graph for serving at.')
    parser.add_argument(
        '--combine_dense',
        action='store_true',
        help='Combine dense layers into a single computational node.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)
