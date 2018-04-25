import argparse
import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import tag_constants

# TODOs: parameterize (batch size, export dir, filename), remove 'labels' from model 
# Note: if you encounter an error with libcupti.so.9.0, just append /usr/local/cuda/extras/CUPTI/lib64 to your LD_LIBRARY_PATH
# Note: INT8 probably won't work

flags = None

# Location of the model to be timed
export_dir = "models/model_serving/"

# Filenames for intermediate models to be saved at
graph_file = "conv_ensemble.pb"
fp32_file = "conv_ensemble_TRTFP32.pb"
fp16_file = "conv_ensemble_TRTFP16.pb"
int8_calib_file = "conv_ensemble_TRTINT8Calib.pb"
int8_infer_file = "conv_ensemble_TRTINT8.pb"

# Set to True to enable different precision optimizations
# native = True
# FP32 = True
# FP16 = False
# INT8 = False

gpu_fraction = 0.75
# batch_size = 64
# num_loops = 10000

# Maximum GPU memory size available for TensorRT
wsize = 1 << 30 #(1 - gpu_fraction) * 10000000000

def getGraph():
  with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
    graph_def = meta_graph_def.graph_def
    with gfile.FastGFile(graph_file, 'wb') as f:
      f.write(graph_def.SerializeToString())
    return graph_def

def printStats(graphName, timings, batch_size):
  if timings is None:
    return
  times = np.array(timings)
  speeds = batch_size / times
  avgTime = np.mean(timings)
  avgSpeed = batch_size / avgTime
  stdTime = np.std(timings)
  stdSpeed = np.std(speeds)
  print("images/s : %.1f +/- %.1f, s/batch: %.5f +/- %.5f" % (avgSpeed, stdSpeed, avgTime, stdTime))
  print("RES, %s, %s, %.2f, %.2f, %.5f, %.5f" % (graphName, batch_size, avgSpeed, stdSpeed, avgTime, stdTime))

def getFP32(batch_size=64, workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getGraph(), [ "classes"],
                                           max_batch_size = batch_size,
                                           max_workspace_size_bytes = workspace_size,
                                           precision_mode = "FP32")  # Get optimized graph
  with gfile.FastGFile(fp32_file, 'wb') as f:
    f.write(trt_graph.SerializeToString())
  writer = tf.summary.FileWriter("logs/32")
  writer.add_graph(trt_graph)
  return trt_graph

def getFP16(batch_size=64, workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getGraph(), [ "classes"],
                                           max_batch_size = batch_size,
                                           max_workspace_size_bytes = workspace_size,
                                           precision_mode = "FP16")  # Get optimized graph
  with gfile.FastGFile(fp16_file, 'wb') as f:
    f.write(trt_graph.SerializeToString())
  writer = tf.summary.FileWriter("logs/16")
  writer.add_graph(trt_graph)
  return trt_graph

def getINT8CalibGraph(batch_size=64, workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getGraph(), [ "classes"],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="INT8")  # calibration
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
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_fraction)
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
    out = tf.import_graph_def(
            graph_def = gdef,
            input_map = {"input": next_element},
            return_elements = [ "labels"]
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
      valt = sess.run(outlist, feed_dict = {"import/labels:0": np.random.randint(10, size = 64)})
    tf.logging.info("Warmup done. Starting real timing")
    num_iters = 50
    for i in range(num_loops):
      tstart = time.time()
      for k in range(num_iters):
        val = sess.run(outlist, feed_dict = {"import/labels:0": np.random.randint(10, size = 64)})
      timings.append((time.time()-tstart)/float(num_iters))
      # print("iter ", i, " ", timings[-1])
    comp = sess.run(tf.reduce_all(tf.equal(val[0], valt[0])))
    print("Comparison = ", comp)
    sess.close()
    tf.logging.info("Timing loop done!")
    return timings, comp, val[0], None

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--verbose',
      default=False,
      action='store_true',
      help='Print step information more often')
  parser.add_argument(
      '--native',
      default=True,
      type=bool,
      help='PLACEHOLDER')
  parser.add_argument(
      '--FP32',
      default=True,
      type=bool,
      help='PLACEHOLDER')
  parser.add_argument(
      '--FP16',
      default=True,
      type=bool,
      help='PLACEHOLDER')
  parser.add_argument(
      '--INT8',
      default=False,
      type=bool,
      help='PLACEHOLDER')
  parser.add_argument(
      '--batch_size',
      default=64,
      type=int,
      help='Specify batch size for evaluation')
  parser.add_argument(
      '--num_loops',
      default=1000,
      type=int,
      help='PLACEHOLDER')
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
    default='models/model.ckpt',
    type=str,
    help='Name of prefix to save model checkpoints under.')
  parser.add_argument(
      '--serving_model_path',
      default='models/model_serving/',
      type=str,
      help='Name of prefix to save model graph for serving at.')
  FLAGS, _ = parser.parse_known_args()

  dummy_input = np.random.random_sample((FLAGS.batch_size, 28, 28, 1))

  if FLAGS.native:
    timings, comp, valnative, mdstats = timeGraph(getGraph(), FLAGS.batch_size,
                                        FLAGS.num_loops, dummy_input)
    writer = tf.summary.FileWriter("logs/native")
    writer.add_graph(getGraph())
    printStats("Native", timings, FLAGS.batch_size)
    printStats("NativeRS", mdstats, FLAGS.batch_size)
  if FLAGS.FP32:
    timings, comp, valfp32, mdstats = timeGraph(getFP32(FLAGS.batch_size, wsize), FLAGS.batch_size, FLAGS.num_loops,
                                      dummy_input)
    printStats("TRT-FP32", timings, FLAGS.batch_size)
    printStats("TRT-FP32RS", mdstats, FLAGS.batch_size)
  if FLAGS.FP16:
    timings, comp, valfp16, mdstats = timeGraph(getFP16(FLAGS.batch_size, wsize), FLAGS.batch_size, FLAGS.num_loops,
                                      dummy_input)
    printStats("TRT-FP16", timings, FLAGS.batch_size)
    printStats("TRT-FP16RS", mdstats, FLAGS.batch_size)
  if FLAGS.INT8:
    calibGraph = getINT8CalibGraph(FLAGS.batch_size, wsize)
    print("Running Calibration")
    timings, comp, _, mdstats = timeGraph(calibGraph, FLAGS.batch_size, 1, dummy_input)
    print("Creating inference graph")
    int8Graph = getINT8InferenceGraph(calibGraph)
    del calibGraph
    timings, comp, valint8, mdstats = timeGraph(int8Graph, FLAGS.batch_size,
                                      FLAGS.num_loops, dummy_input)
    printStats("TRT-INT8", timings, FLAGS.batch_size)
    printStats("TRT-INT8RS", mdstats, FLAGS.batch_size)
