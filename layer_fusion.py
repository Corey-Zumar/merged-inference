import tensorflow as tf

# Code that attempts to find candidate layers to merge together for inference time
# worthy experiment: How does cuBLAS matrix multiply performance scale with matrix size?
# worhty experiment: is a convolution way more expensive than a matrix multiply?
# Hard constraints: GPU multiply unit size (12GB - (1/3) * 12 = 8GB), multiplication precision
# Soft constraints: For small models, we could probably merge quite a few kernels

# TODO: Remove layers with unused output nodes

# vertical layer fusion is probably just making a bunch of ops work in a single CUDA program

# horizontal layer fusion most optimal when you take the same source tensor

# get colocated models
g1 = tf.Graph()
g2 = tf.Graph()
export_dir1 = 'tensorflow_models/models/model_serving'
export_dir2 = 'models/model2.ckpt'
with tf.Session(graph=g1) as sess1:
    tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], export_dir1)
    print(sess1.graph.get_operations())

# with tf.Session(graph=g2) as sess:
#     tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir2)
#     print(sess2.graph.get_operations())

global_unique_tensors = {'input'}

# cost estimates of each layer will make it easier to figure out which computations are easiest to merge.
# compute average or hardcode
# iterate through model graphs and categorize them

# iterate through categorized layers and evaluate them for combination

# 
