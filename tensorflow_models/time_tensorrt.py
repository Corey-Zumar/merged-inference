import numpy as np
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import tag_constants
import time
from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.python.framework import ops as ops
from tensorflow.python.framework import importer as importer
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

# TODOs: fix imports, remove run_graph, add back FP16 etc, parameterize batch size, export dir, filename
# Note: if you encounter an error with libcupti.so.9.0, just append /usr/local/cuda/extras/CUPTI/lib64 to your LD_LIBRARY_PATH

export_dir = "models/model_serving/"

def read_tensor_from_image_file(file_name, input_height=28, input_width=28,
                                input_mean=0, input_std=255):
  """ Read a jpg image file and return a tensor """
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  image_reader = tf.image.decode_png(file_reader, channels = 1,
                                       name='jpg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50)))
  result = sess.run([normalized,tf.transpose(normalized,perm=(0,3,1,2))])
  del sess

  return result

def getGraph():
  with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
    graph_def = meta_graph_def.graph_def
    with gfile.FastGFile("conv_ensemble.pb",'wb') as f:
      f.write(graph_def.SerializeToString())
    return graph_def

def printStats(graphName,timings,batch_size):
  if timings is None:
    return
  times=np.array(timings)
  speeds=batch_size / times
  avgTime=np.mean(timings)
  avgSpeed=batch_size/avgTime
  stdTime=np.std(timings)
  stdSpeed=np.std(speeds)
  print("images/s : %.1f +/- %.1f, s/batch: %.5f +/- %.5f"%(avgSpeed,stdSpeed,avgTime,stdTime))
  print("RES, %s, %s, %.2f, %.2f, %.5f, %.5f"%(graphName,batch_size,avgSpeed,stdSpeed,avgTime,stdTime))

def getFP32(batch_size=128,workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getGraph(), [ "classes"],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="FP32")  # Get optimized graph
  with gfile.FastGFile("conv_ensemble_TRTFP32.pb",'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def run_graph(gdef, dumm_inp):
  """Run given graphdef once."""
  gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
  ops.reset_default_graph()
  g = ops.Graph()
  with g.as_default():
    inp, out = importer.import_graph_def(
        graph_def=gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
  with csess.Session(
      config=cpb2.ConfigProto(gpu_options=gpu_options), graph=g) as sess:
    val = sess.run(out, {inp: dumm_inp})
  return val

def timeGraph(gdef,batch_size=128,num_loops=100,dummy_input=None,timelineName=None):
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
  tf.reset_default_graph()
  g = tf.Graph()
  if dummy_input is None:
    dummy_input = np.random.random_sample((batch_size, 28, 28, 1))
  outlist=[]
  # outlist=["classes"]
  with g.as_default():
    inc=tf.constant(dummy_input, dtype=tf.float32)
    dataset=tf.data.Dataset.from_tensors(inc)
    dataset=dataset.repeat()
    iterator=dataset.make_one_shot_iterator()
    next_element=iterator.get_next()
    out = tf.import_graph_def(
      graph_def=gdef,
      input_map={"input":next_element},
      return_elements=[ "labels"]
    )
    out = out[0].outputs[0]
    outlist.append(out)
    
  timings=[]
  
  with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf.logging.info("Starting Warmup cycle")
    def mergeTraceStr(mdarr):
      tl=timeline.Timeline(mdarr[0][0].step_stats)
      ctf=tl.generate_chrome_trace_format()
      Gtf=json.loads(ctf)
      deltat=mdarr[0][1][1]
      for md in mdarr[1:]:
        tl=timeline.Timeline(md[0].step_stats)
        ctf=tl.generate_chrome_trace_format()
        tmp=json.loads(ctf)
        deltat=0
        Gtf["traceEvents"].extend(tmp["traceEvents"])
        deltat=md[1][1]
        
      return json.dumps(Gtf,indent=2)
    rmArr=[[tf.RunMetadata(),0] for x in range(20)]
    if timelineName:
      if gfile.Exists(timelineName):
        gfile.Remove(timelineName)
      ttot=int(0)
      tend=time.time()
      for i in range(20):
        tstart=time.time()
        valt = sess.run(outlist,options=run_options,run_metadata=rmArr[i][0])
        tend=time.time()
        rmArr[i][1]=(int(tstart*1.e6),int(tend*1.e6))
      with gfile.FastGFile(timelineName,"a") as tlf:
        tlf.write(mergeTraceStr(rmArr))
    else:
      for i in range(20):
        valt = sess.run(outlist)
    tf.logging.info("Warmup done. Starting real timing")
    num_iters=50
    for i in range(num_loops):
      tstart=time.time()
      for k in range(num_iters):
        val = sess.run(outlist)
      timings.append((time.time()-tstart)/float(num_iters))
      print("iter ",i," ",timings[-1])
    comp=sess.run(tf.reduce_all(tf.equal(val[0],valt[0])))
    print("Comparison=",comp)
    sess.close()
    tf.logging.info("Timing loop done!")
    return timings,comp,val[0],None

batch_size = 64
num_loops = 20
workspace_size = 1 << 10
wsize = workspace_size << 20

imageName="grace_hopper.jpg"
t = read_tensor_from_image_file(imageName,
                              input_height=28,
                              input_width=28,
                              input_mean=0,
                              input_std=1.0)
# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# t = tf.reshape(mnist.train.images[0], [28, 28, 1])
# print(t.shape)

tshape = list(t[0].shape)
tshape[0] = batch_size
tnhwcbatch = np.tile(t[0],(batch_size,1,1,1))
dummy_input = tnhwcbatch


timelineName = "FP32Timeline.json"
# run_graph(getFP32(batch_size,wsize), dummy_input)
timings,comp,valfp32,mdstats=timeGraph(getFP32(batch_size,wsize),batch_size,num_loops,
                               dummy_input,timelineName)
printStats("TRT-FP32",timings,batch_size)
printStats("TRT-FP32RS",mdstats,batch_size)

