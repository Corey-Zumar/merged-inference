# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import tag_constants

# TODOs: parameterize filenames, remove 'labels' from model 
# Note: if you encounter an error with libcupti.so.9.0, just append /usr/local/cuda/extras/CUPTI/lib64 to your LD_LIBRARY_PATH
# Note: INT8 probably won't work

flags = None

# Filter out WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Maximum GPU memory size available for TensorRT
wsize = 1 << 30 #(1 - FLAGS.gpu_fraction) * 10000000000

def getGraph():
  with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], FLAGS.model_path)
    graph_def = meta_graph_def.graph_def
    with gfile.FastGFile(graph_file, 'wb') as f:
      f.write(graph_def.SerializeToString())
    return graph_def

def printStats(graphName, timings, batch_size):
  times = np.array(timings)
  speeds = batch_size / times
  avgTime = np.mean(timings)
  avgSpeed = batch_size / avgTime
  stdTime = np.std(timings)
  stdSpeed = np.std(speeds)
  print(graphName)
  print("images/s : %.1f +/- %.1f, μs/batch: %.5f +/- %.5f" % (avgSpeed, stdSpeed, avgTime * 1000000, stdTime * 1000000))
  # print("RES, %s, %s, %.2f, %.2f, %.5f, %.5f" % (graphName, batch_size, avgSpeed, stdSpeed, avgTime, stdTime))

def getFP32(batch_size=64, workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getGraph(), [ "classes"],
                                           max_batch_size = batch_size,
                                           max_workspace_size_bytes = workspace_size,
                                           precision_mode = "FP32")
  with gfile.FastGFile(fp32_file, 'wb') as f:
    f.write(trt_graph.SerializeToString())
  if FLAGS.logging_enabled:
    writer = tf.summary.FileWriter(FLAGS.log_dir_prefix + "32")
    writer.add_graph(trt_graph)
  return trt_graph

def getFP16(batch_size=64, workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getGraph(), [ "classes"],
                                           max_batch_size = batch_size,
                                           max_workspace_size_bytes = workspace_size,
                                           precision_mode = "FP16")
  with gfile.FastGFile(fp16_file, 'wb') as f:
    f.write(trt_graph.SerializeToString())
  if FLAGS.logging_enabled:
    writer = tf.summary.FileWriter(FLAGS.log_dir_prefix + "16")
    writer.add_graph(trt_graph)
  return trt_graph

def getINT8CalibGraph(batch_size=64, workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getGraph(), [ "classes"],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="INT8")
  with gfile.FastGFile(int8_calib_file,'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def getINT8InferenceGraph(calibGraph):
  trt_graph = trt.calib_graph_to_infer_graph(calibGraph)
  with gfile.FastGFile(int8_infer_file, 'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def timeGraph(gdef, batch_size=64, num_loops=100, dummy_input=None):
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_fraction)
  tf.reset_default_graph()
  g = tf.Graph()
  if dummy_input is None:
    dummy_input = np.random.random_sample((batch_size, 28, 28, 1))
  outlist = []
  with g.as_default():
    inc = tf.constant(dummy_input, dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensors(inc)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    out = None
    if FLAGS.conv_or_dense:
      out = tf.import_graph_def(
              graph_def = gdef,
              input_map = {"input": next_element},
              return_elements = ["labels"]
      )
    else:
      out = tf.import_graph_def(
              graph_def = gdef,
              input_map = {"mnist_inputs00" + FLAGS.num_ensembles: next_element},
              return_elements = ["mnist_labels"]
      )
    out = out[0].outputs[0]
    outlist.append(out)
    
  timings = []
  
  with tf.Session(graph = g, config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf.logging.info("Starting Warmup cycle")
    rmArr = [[tf.RunMetadata(), 0] for x in range(20)]
    for i in range(20):
      valt = None
      if FLAGS.conv_or_dense:
        valt = sess.run(outlist, feed_dict = {"import/labels:0": dummy_labels})
      else:
        valt = sess.run(outlist, feed_dict = {"import/mnist_labels:0": dummy_labels})   
    tf.logging.info("Warmup done. Starting real timing")
    num_iters = 50
    for i in range(num_loops):
      tstart = time.time()
      for k in range(num_iters):
        if FLAGS.conv_or_dense:
          val = sess.run(outlist, feed_dict = {"import/labels:0": dummy_labels})
        else:
          val = sess.run(outlist, feed_dict = {"import/mnist_labels:0": dummy_labels})
      timings.append((time.time() - tstart) / float(num_iters))
      # print("iter ", i, " ", timings[-1])
    sess.close()
    tf.logging.info("Timing loop done!")
    return timings, val[0]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--native',
      default=True,
      type=bool,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--FP32',
      default=True,
      type=bool,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--FP16',
      default=True,
      type=bool,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--INT8',
      default=False,
      type=bool,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--batch_size',
      default=64,
      type=int,
      help='specify batch size for evaluation',
      metavar='')
  parser.add_argument(
      '--num_loops',
      default=1000,
      type=int,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--seed',
      default=0,
      type=int,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--gpu_fraction',
      default=0.75,
      type=float,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--logging_enabled',
      default=True,
      type=bool,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--num_ensembles',
      default=4,
      type=int,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--log_dir_prefix',
      default='logs/',
      type=str,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--conv_or_dense',
      default=True,
      type=bool,
      help='PLACEHOLDER',
      metavar='')
  parser.add_argument(
      '--model_path',
      default='models/model_serving/',
      type=str,
      help='name of prefix to save model graph for serving at',
      metavar='')
  FLAGS, _ = parser.parse_known_args()

  prefix = None
  if FLAGS.conv_or_dense:
    prefix = "conv"
  else:
    prefix = "dense"

  # Filenames for intermediate models to be saved at
  graph_file = prefix + "_ensemble.pb"
  fp32_file = prefix + "_ensemble_TRTFP32.pb"
  fp16_file = prefix + "_ensemble_TRTFP16.pb"
  int8_calib_file = prefix + "_ensemble_TRTINT8Calib.pb"
  int8_infer_file = prefix + "_ensemble_TRTINT8.pb"

  np.random.seed(FLAGS.seed)
  dummy_input = None
  if FLAGS.conv_or_dense:
    dummy_input = np.random.random_sample((FLAGS.batch_size, 28, 28, 1))
  else:
    dummy_input = np.random.random_sample((FLAGS.batch_size, 784))
  dummy_labels = np.random.randint(10, size=64)
  
  print("batch_size = %i" % FLAGS.batch_size)
  print("num_loops = %i" % FLAGS.num_loops)
  print("gpu_fraction = %.2f" % FLAGS.gpu_fraction)
  print("seed = %i" % FLAGS.seed)

  if FLAGS.native:
    timings, valnative = timeGraph(getGraph(), FLAGS.batch_size,
                                        FLAGS.num_loops, dummy_input)
    if FLAGS.logging_enabled:
      writer = tf.summary.FileWriter(FLAGS.log_dir_prefix + "native")
      writer.add_graph(getGraph())
    printStats("Native", timings, FLAGS.batch_size)
  if FLAGS.FP32:
    timings, valfp32 = timeGraph(getFP32(FLAGS.batch_size, wsize), FLAGS.batch_size, FLAGS.num_loops,
                                      dummy_input)
    printStats("TRT-FP32", timings, FLAGS.batch_size)
  if FLAGS.FP16:
    timings, valfp16 = timeGraph(getFP16(FLAGS.batch_size, wsize), FLAGS.batch_size, FLAGS.num_loops,
                                      dummy_input)
    printStats("TRT-FP16", timings, FLAGS.batch_size)
  if FLAGS.INT8:
    calibGraph = getINT8CalibGraph(FLAGS.batch_size, wsize)
    print("Running Calibration")
    timings, _ = timeGraph(calibGraph, FLAGS.batch_size, 1, dummy_input)
    print("Creating inference graph")
    int8Graph = getINT8InferenceGraph(calibGraph)
    del calibGraph
    timings, valint8 = timeGraph(int8Graph, FLAGS.batch_size,
                                      FLAGS.num_loops, dummy_input)
    printStats("TRT-INT8", timings, FLAGS.batch_size)
