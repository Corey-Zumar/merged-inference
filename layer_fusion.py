from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from google.protobuf import message
from google.protobuf import text_format

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.contrib import graph_editor as ge
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
import tensorflow as tf
import copy
from tensorflow.core.framework import graph_pb2
# Code that attempts to find candidate layers to merge together for inference time
# worthy experiment: How does cuBLAS matrix multiply performance scale with matrix size?
# worthy experiment: is a convolution way more expensive than a matrix multiply?
# Hard constraints: GPU multiply unit size (12GB - (1/3) * 12 = 8GB), multiplication precision
# Soft constraints: For small models, we could probably merge quite a few kernels

# TODO: Remove layers with unused output nodes

# vertical layer fusion is probably just making a bunch of ops work in a single CUDA program

# horizontal layer fusion most optimal when you take the same source tensor


def is_irrelevant_op(op):
    op_name = '/'.join(op.name.split('/')[1:])
    filtered_start_terms = ['GradientDescent', 'save', 'gradients', 'init', 'sparse_softmax_cross_entropy_loss', 'accuracy']
    for t in filtered_start_terms:
        if op_name.startswith(t):
            return True
    filtered_name_tokens = ['Initializer', 'Assign']
    tokens = op.name.split('/')
    for tok in tokens:
        if tok in filtered_name_tokens:
            return True
    return False


def is_equivalent_op(o1, o2):
    if o1.type != o2.type:
        print('Type failed')
        return False
    if o1.device != o2.device:
        print('Device failed')
        return False
    # checks things like DTYPE and shape
    if set(o1.node_def.attr.keys()) != set(o2.node_def.attr.keys()):
        print('Failed node def', set(o1.node_def.attr.keys()), set(o2.node_def.attr.keys()))
        return False
    for k in o1.node_def.attr.keys():
        if o1.node_def.attr[k] != o2.node_def.attr[k]:
            return False
    return True


def parse_saved_model(export_dir):
    """Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.

    Args:
    export_dir: Directory containing the SavedModel file.

    Returns:
    A `SavedModel` protocol buffer.

    Raises:
    IOError: If the file does not exist, or cannot be successfully parsed.
    """
    # Build the path to the SavedModel in pbtxt format.
    path_to_pbtxt = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
    # Build the path to the SavedModel in pb format.
    path_to_pb = os.path.join(
      compat.as_bytes(export_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))

    # Parse the SavedModel protocol buffer.
    saved_model = saved_model_pb2.SavedModel()
    if file_io.file_exists(path_to_pb):
        try:
            file_content = file_io.FileIO(path_to_pb, "rb").read()
            saved_model.ParseFromString(file_content)
            return saved_model
        except message.DecodeError as e:
          raise IOError("Cannot parse file %s: %s." % (path_to_pb, str(e)))
    elif file_io.file_exists(path_to_pbtxt):
        try:
            file_content = file_io.FileIO(path_to_pbtxt, "rb").read()
            text_format.Merge(file_content.decode("utf-8"), saved_model)
            return saved_model
        except text_format.ParseError as e:
            raise IOError("Cannot parse file %s: %s." % (path_to_pbtxt, str(e)))
    else:
        raise IOError("SavedModel file does not exist at: %s/{%s|%s}" %
                  (export_dir,
                   constants.SAVED_MODEL_FILENAME_PBTXT,
                   constants.SAVED_MODEL_FILENAME_PB))


def metagraph_from_saved_model(saved_model, tags):
    for meta_graph_def in saved_model.meta_graphs:
        # HACK: Set to 1.
        if set(meta_graph_def.meta_info_def.tags) == set(tags):
            return meta_graph_def
    raise Exception("No Metagraph to import.")


def get_asset_tensors(export_dir, meta_graph_def_to_load):
    """Gets the asset tensors, if defined in the meta graph def to load.

    Args:
        export_dir: Directory where the SavedModel is located.
        meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.

    Returns:
        A dictionary of asset tensors, keyed by the name of the asset tensor. The
        value in the map corresponds to the absolute path of the asset file.
  """
    # Collection-def that may contain the assets key.
    collection_def = meta_graph_def_to_load.collection_def

    asset_tensor_dict = {}
    if constants.ASSETS_KEY in collection_def:
    # Location of the assets for SavedModel.
        assets_directory = os.path.join(
            compat.as_bytes(export_dir),
            compat.as_bytes(constants.ASSETS_DIRECTORY))
        assets_any_proto = collection_def[constants.ASSETS_KEY].any_list.value
        # Process each asset and add it to the asset tensor dictionary.
        for asset_any_proto in assets_any_proto:
            asset_proto = meta_graph_pb2.AssetFileDef()
            asset_any_proto.Unpack(asset_proto)
            asset_tensor_dict[asset_proto.tensor_info.name] = os.path.join(
              compat.as_bytes(assets_directory),
              compat.as_bytes(asset_proto.filename))
    return asset_tensor_dict


def uses_any_output_tensor(op, prev_outs):
    for t in prev_outs:
        if t in op.inputs:
            return True
    return False

# get colocated models
g1 = tf.Graph()
g2 = tf.Graph()
g3 = tf.Graph()
export_dir1 = 'tensorflow_models/models/dense1/1/'
input1 = 'mnist_inputs'
input2 = 'mnist_inputs_v2'
export_dir2 = 'tensorflow_models/models/dense4/1/'

with tf.Session(graph=g1) as sess1:
    tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], export_dir1, import_scope='m1')
    tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], export_dir2, import_scope='m2')
    all_ops = sess1.graph.get_operations()
    print('ALL OPS:', all_ops)
    m1_ops = [op for op in all_ops if op.name.split('/')[0] == 'm1']
    m2_ops = [op for op in all_ops if op.name.split('/')[0] == 'm2']
    filtered_ops_1 = [op for op in m1_ops if not is_irrelevant_op(op)]
    # print(len(sess1.graph.get_collection(tf.GraphKeys.)))
    source_ops_1 = [op for op in filtered_ops_1 if len([i.name for i in op.inputs]) == 0]
    filtered_ops_2 = [op for op in m2_ops if not is_irrelevant_op(op)]
    source_ops_2 = [op for op in filtered_ops_2 if len([i.name for i in op.inputs]) == 0]
    s_op_1, s_op_2 = None, None
    for op in source_ops_1:
        if op.name == input1:
            s_op_1 = op
    for op in source_ops_2:
        if op.name == input2:
            s_op_2 = op
    if s_op_1 is None:
        for op in source_ops_1:
            if 'input' in op.name:
                s_op_1 = op
                break
    if s_op_1 is None:
        raise Exception("No input op found for graph 1")
    if s_op_2 is None:
        for op in source_ops_2:
            if 'input' in op.name:
                s_op_2 = op
                break
    if s_op_2 is None:
        raise Exception("No input op found for graph 2")
    is_match = True
    cur_ops_1, cur_ops_2 = [s_op_1], [s_op_2]
    traj_1, traj_2 = [s_op_1], [s_op_2]
    while is_match:
        next_op_1 = [op for op in filtered_ops_1 if uses_any_output_tensor(op, cur_ops_1[0].outputs)]
        next_op_2 = [op for op in filtered_ops_2 if uses_any_output_tensor(op, cur_ops_2[0].outputs)]
        print('Next Op 1:', next_op_1)
        print('Next Op 2:', next_op_2)
        matched_ops_1, matched_ops_2 = [], []
        while len(next_op_1) != 0:
            for i, op in enumerate(next_op_2):
                print('Comparing', op.name, 'with', next_op_1[0].name)
                if is_equivalent_op(next_op_1[0], op):
                    print('found match')
                    matched_ops_1.append(next_op_1[0])
                    matched_ops_2.append(op)
                    next_op_1 = next_op_1[1:]
                    continue
            is_match = False
            break
        cur_ops_1 = cur_ops_1[1:]
        cur_ops_2 = cur_ops_2[1:]
        traj_1.extend(matched_ops_1)
        traj_2.extend(matched_ops_2)
        cur_ops_1.extend(matched_ops_1)
        cur_ops_2.extend(matched_ops_2)
    # mergable operations are traj_1.
    print('Trajectory1:', traj_1)
    print('Trajectory2:', traj_2)
    print('Outs:', traj_1[0].outputs)
    input_stack = tf.concat(values=[traj_1[0].outputs[0], traj_2[0].outputs[0]], axis=0, name='new_input')
    print('Input Vector', input_stack)
    print('T1Inputs:', [x for x in traj_1[1].inputs])
    # for all the preprocessing ops, look if there's an input tensor edge.
    for o in traj_1[1:]:
        print('Doing subs for', o.name)
        if traj_1[0].outputs[0].name in [x.name for x in o.inputs]:
            print('Before:', [x.name for x in o.inputs])
            new_inputs = [input_stack] + [x for x in o.inputs if x.name != traj_1[0].outputs[0].name]
            ge.swap_inputs(o, new_inputs)
            print('Subbed:', [x.name for x in o.inputs])
    post_processing_g1 = [op for op in filtered_ops_1 if op not in traj_1]
    post_processing_g2 = [op for op in filtered_ops_2 if op not in traj_2]
    new_g_def = graph_pb2.GraphDef()
    new_g_def.node.extend([copy.deepcopy(s_op_2.node_def)])
    new_g_def.node.extend([copy.deepcopy(s_op_1.node_def)])
    concat_ops = [op for op in sess1.graph.get_operations() if 'new_input' in op.name]
    print(concat_ops)
    tensor_to_split = traj_1[-1]
    print('SUB OUTS:', tensor_to_split.outputs)
    s1, s2 = tf.shape(traj_1[0].outputs[0])[0], tf.shape(traj_2[0].outputs[0])[0]
    split_1, split_2 = tf.split(tensor_to_split.outputs[0], [s1, s2], name='m1/split')
    split_ops = [op for op in sess1.graph.get_operations()
                 if 'split' in op.name or
                 (op not in post_processing_g1
                  and op not in post_processing_g2
                  and op not in traj_1
                  and op not in traj_2
                  and op not in concat_ops
                  and not is_irrelevant_op(op))]
    print('SPLIT OPS:', split_ops)
    for o in post_processing_g1:
        print('Doing subs for', o.name)
        if traj_1[-1].outputs[0].name in [x.name for x in o.inputs]:
            print('Before:', [x.name for x in o.inputs])
            new_inputs = [split_1] + [x for x in o.inputs if x.name != traj_1[-1].outputs[0].name]
            ge.swap_inputs(o, new_inputs)
            print('Subbed:', [x.name for x in o.inputs])
    for o in post_processing_g2:
        print('Doing subs for', o.name)
        print('Subbing:', traj_2[-1].outputs[0].name)
        if traj_2[-1].outputs[0].name in [x.name for x in o.inputs]:
            print('Before:', [x.name for x in o.inputs])
            new_inputs = [split_2] + [x for x in o.inputs if x.name != traj_2[-1].outputs[0].name]
            ge.swap_inputs(o, new_inputs)
            print('Subbed:', [x.name for x in o.inputs])
    print('**********')
    for o in post_processing_g2:
        print(o.name, ':', [x.name for x in o.inputs])
    print('o1')
    print('**********')
    for o in post_processing_g1:
        print(o.name, ':', [x.name for x in o.inputs])
    for n in concat_ops:
        new_g_def.node.extend([copy.deepcopy(n.node_def)])
    for n in split_ops:
        new_g_def.node.extend([copy.deepcopy(n.node_def)])
    for n in traj_1[1:]:
        new_g_def.node.extend([copy.deepcopy(n.node_def)])
    for n in post_processing_g1:
        new_g_def.node.extend([copy.deepcopy(n.node_def)])
    for n in post_processing_g2:
        new_g_def.node.extend([copy.deepcopy(n.node_def)])
    with tf.Session(graph=g3) as sess3:
        # saved_m = parse_saved_model(export_dir1)
        # mgf = metagraph_from_saved_model(saved_m, [tf.saved_model.tag_constants.SERVING])
        # tensor_dict = get_asset_tensors(export_dir1, mgf)
        # print(tensor_dict)
        tf.import_graph_def(new_g_def)
        tf.summary.FileWriter('tensorflow_models/models/dense/summary_3/', sess3.graph)



# with tf.Session(graph=g2) as sess:
#     tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir2)
#     print(sess2.graph.get_operations())

# cost estimates of each layer will make it easier to figure out which computations are easiest to merge.
# compute average or hardcode
# iterate through model graphs and categorize them

# iterate through categorized layers and evaluate them for combination

# 
