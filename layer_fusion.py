import tensorflow as tf

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
