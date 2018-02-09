import tensorflow as tf
import numpy as np
import argparse
import logging

from datetime import datetime

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)


def eval_sequential(gpu_num, num_trials):
    logger.info("Running {nt} sequential trials on gpu {gn}".format(
        nt=num_trials, gn=gpu_num))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device("/gpu:{}".format(gpu_num)):
        t_inputs = tf.placeholder(tf.float32, [None, 256 * 256])
        t_mat_a = tf.random_normal([256 * 256, 100], dtype=tf.float32)
        t_mat_b = tf.random_normal([256 * 256, 200], dtype=tf.float32)
        t_output_a = tf.matmul(t_inputs, t_mat_a)
        t_output_b = tf.matmul(t_inputs, t_mat_b)

    sess.run(tf.global_variables_initializer())

    latencies = []
    for _ in range(num_trials):
        inputs = np.random.rand(10, 256 * 256)
        feed_dict = {t_inputs: inputs}
        begin = datetime.now()
        output_a, output_b = sess.run(
            [t_output_a, t_output_b], feed_dict=feed_dict)
        end = datetime.now()
        latency = (end - begin).total_seconds()
        latencies.append(latency)
        logger.info("Latency: {} seconds".format(latency))


def eval_merged(gpu_num, num_trials):
    logger.info("Running {nt} merged trials on gpu {gn}".format(
        nt=num_trials, gn=gpu_num))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device("/gpu:{}".format(gpu_num)):
        t_inputs = tf.placeholder(tf.float32, [None, 256 * 256])
        t_mat = tf.random_normal([256 * 256, 300], dtype=tf.float32)
        t_output = tf.matmul(t_inputs, t_mat)

    sess.run(tf.global_variables_initializer())

    latencies = []
    for _ in range(num_trials):
        inputs = np.random.rand(10, 256 * 256)
        feed_dict = {t_inputs: inputs}
        begin = datetime.now()
        output = sess.run(t_output, feed_dict=feed_dict)
        end = datetime.now()
        latency = (end - begin).total_seconds()
        latencies.append(latency)
        logger.info("Latency: {} seconds".format(latency))


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark matrix multiplication operations')
    parser.add_argument(
        "-g",
        "--gpu_num",
        type=int,
        default=0,
        help="The number of the GPU on which to execute the experiment")
    parser.add_argument(
        "-t",
        "--num-trials",
        type=int,
        default=50,
        help="The number of trials to conduct")
    parser.add_argument(
        "-s",
        "--sequential",
        action='store_true',
        default=False,
        help="If specified, runs the sequential multiplication experiment")
    parser.add_argument(
        "-m",
        "--merged",
        action='store_true',
        default=False,
        help="If specified, runs the merged multiplication experiment")

    args = parser.parse_args()

    if args.sequential:
        eval_sequential(args.gpu_num, args.num_trials)
    elif args.merged:
        eval_merged(args.gpu_num, args.num_trials)
    else:
        raise Exception("Either the -m/--merged or -s/--sequential \
                flag must be specified!")


if __name__ == "__main__":
    main()
