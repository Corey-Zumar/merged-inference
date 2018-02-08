import sys
import os
import tensorflow as tf
import numpy as np
import time
import logging

from datetime import datetime

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

def initialize_model(model, gpu_num=0):
    with tf.device("/gpu:{}".format(gpu_num)):
        model.create_graph()

def create_session():
    return tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

def eval_model(model, batch_size, gpu_num=0, sess=None):
    if not sess:
        sess = create_sesion()
    initialize_model(model, gpu_num)
    inputs = model.get_inputs(num_inputs=batch_size * 2)
    idxs = np.random.randint(len(inputs), size=batch_size)
    feed_dict = {
        model.get_input_tensor() : inputs[idxs]
    }
    before = datetime,.now()
    outputs = sess.run(model.get_output_tensor(), feed_dict=feed_dict)
    after = datetime.now()
    latency = (after - before).total_seconds()
    logger.info("{mod} Latency: {lat}".format(mod=model.get_name(), lat=latency))
    tf.reset_default_graph()
    return latency

def eval_sequential(models, batch_size, gpu_num=0):
    sess = create_session()
    total_latency = 0
    for model in models:
        latency = eval_model(model=model,
                             batch_size=batch_size,
                             gpu_num=gpu_num)
        total_latency += latency

    logger.info("Total Latency: {}".format(total_latency))
