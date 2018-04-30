import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time
import argparse
import json

from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from torch.autograd import Variable
from datetime import datetime

OUTPUT_KEY_NODE_CONFIG = "node_config"
OUTPUT_KEY_STATS = "stats"
OUTPUT_KEY_TRIAL_LENGTH = "trial_length"
OUTPUT_KEY_NUM_TRIALS = "num_trials"

class NodeConfig:

   def __init__(self, gpu_type, num_gpus, num_pcpus):
      self.gpu_type = gpu_type
      self.num_gpus = num_gpus
      self.num_pcpus = num_pcpus

class StatsManager:

   def __init__(self, batch_size, trial_length):
      self.batch_size = batch_size
      self.trial_length = trial_length
      self.stats = { 
                     "thrus" : [],
                     "batch_preprocessing_latencies" : [], 
                     "batch_resnet_latencies" : [],
                     "batch_alexnet_latencies" : [],
                     "batch_total_latencies" : []
                   }

      self._init_stats()
      
   def _init_stats(self):
      self.trial_start_time = datetime.now()
      self.trial_num_complete = 0

   def add_data(self, batch_size, preprocessing_latency, resnet_latency, alexnet_latency, total_latency):
      self.stats["batch_preprocessing_latencies"].append(preprocessing_latency)
      self.stats["batch_resnet_latencies"].append(resnet_latency)
      self.stats["batch_alexnet_latencies"].append(alexnet_latency)
      self.stats["batch_total_latencies"].append(total_latency)
      self.trial_num_complete += batch_size

      if self.trial_num_complete >= self.trial_length:
         trial_end_time = datetime.now()
         new_thru = float(self.trial_num_complete) / (trial_end_time - self.trial_start_time).total_seconds()
         self.stats["thrus"].append(new_thru)
         
         self._init_stats()

def load_resnet():
    resnet = models.resnet152(pretrained=True)
    if torch.cuda.is_available():
        print("Optimizing model for GPU execution")
        resnet.cuda()
    else:
        raise

    resnet.eval()

    return resnet

def load_alexnet():
    alexnet = models.alexnet(pretrained=True)
    if torch.cuda.is_available():
        print("Optimizing model for GPU execution")
        alexnet.cuda()
    else:
        raise
    
    alexnet.eval()

    return alexnet

def evaluate_sequential_naive(resnet_model, alexnet_model, ensemble_batch_size, num_trials, trial_length):
   preprocess = transforms.Compose(
         [transforms.Scale(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()]
   )

   def preprocess_fn(items):
      preprocessed_items = np.empty(len(items), dtype=object)
      for idx in range(len(items)):
         item = items[idx]
         img_item = Image.fromarray(item, mode="RGB")
         img_item = preprocess(img_item)
         preprocessed_items[idx] = img_item

      return preprocessed_items

   stats_manager = StatsManager(ensemble_batch_size, trial_length) 

   print("Generating batch inputs...") 

   batch_inputs = [np.random.rand(ensemble_batch_size, 3, 224, 224) for _ in range(100)]
   batch_inputs = [i for _ in range(40) for i in batch_inputs]

   print("Running experiment...")

   for idx in xrange(len(batch_inputs)):
      all_items = batch_inputs[idx]
      # These operations would execute in an isolated ResNet container
      t1 = datetime.now()
      preprocessed_resnet = preprocess_fn(all_items)
      resnet_inp_var = Variable(torch.stack(preprocessed_resnet, dim=0)).cuda()
      t2 = datetime.now()
      resnet_model(resnet_inp_var)
      t3 = datetime.now()
     
      # These operations would execute in an isolated AlexNet container (after the previous operations)
      t4 = datetime.now()
      preprocessed_alexnet = preprocess_fn(all_items)
      alexnet_inp_var = Variable(torch.stack(preprocessed_alexnet, dim=0)).cuda()
      t5 = datetime.now()
      alexnet_model(alexnet_inp_var)
      t6 = datetime.now()

      resnet_latency = (t3 - t2).total_seconds()
      alexnet_latency = (t6 - t5).total_seconds()
      preprocessing_latency = (t5 - t4).total_seconds() + (t2 - t1).total_seconds()
      total_latency = (t6 - t1).total_seconds()

      stats_manager.add_data(ensemble_batch_size, preprocessing_latency, resnet_latency, alexnet_latency, total_latency)

      print("TRIAL LATENCY", total_latency)

      if len(stats_manager.stats["thrus"]) > num_trials:
         break

   return stats_manager.stats

def evaluate_sequential_fused(resnet_model, alexnet_model, ensemble_batch_size, num_trials, trial_length):
   preprocess = transforms.Compose(
         [transforms.Scale(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()]
   )

   def preprocess_fn(items):
      preprocessed_items = np.empty(len(items), dtype=object)
      for idx in range(len(items)):
         item = items[idx]
         img_item = Image.fromarray(item, mode="RGB")
         img_item = preprocess(img_item)
         preprocessed_items[idx] = img_item

      return preprocessed_items

   stats_manager = StatsManager(ensemble_batch_size, trial_length) 

   print("Generating batch inputs...") 

   batch_inputs = [np.random.rand(ensemble_batch_size, 3, 224, 224) for _ in range(100)]
   batch_inputs = [i for _ in range(40) for i in batch_inputs]

   print("Running experiment...")

   for idx in xrange(200):
      all_items = batch_inputs[idx]

      # These operations would execute in the same container 
      t0 = datetime.now()
      preprocessed_items = preprocess_fn(all_items)
      inp_var = Variable(torch.stack(preprocessed_items, dim=0)).cuda()
      t1 = datetime.now()

      t2 = datetime.now()
      resnet_model(inp_var)
      t3 = datetime.now()
      alexnet_model(inp_var)
      t4 = datetime.now()

      resnet_latency = (t3 - t2).total_seconds()
      alexnet_latency = (t4 - t3).total_seconds()
      preprocessing_latency = (t1 - t0).total_seconds()
      total_latency = (t4 - t0).total_seconds()

      stats_manager.add_data(ensemble_batch_size, preprocessing_latency, resnet_latency, alexnet_latency, total_latency)

      print("TRIAL LATENCY", total_latency)

      if len(stats_manager.stats["thrus"]) > num_trials:
         break

   return stats_manager.stats
         

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Pytorch multimodel experimentation")
   parser.add_argument('-t', '--trials', type=int, help='The number of experimental trials to run')
   parser.add_argument('-tl', '--trial_length', type=int, help='The length of each experimental trial')
   parser.add_argument('-b', '--batch_size', type=int, help='The ensemble batch size to use')
   parser.add_argument('-o', '--output_path', type=str, help='The path to which to save experimental results')
   parser.add_argument('-n', '--naive', action='store_true', help='If specified, run the ensemble using sequential, separate containers')
   parser.add_argument('-f', '--fused', action='store_true', help='If specified, run the ensemble in a single, "fused" container')

   args = parser.parse_args()

   gpu_type = "tesla-k80"
   num_gpus = 1
   num_pcpus = 1 # Virtual cores 0 and 16

   node_config = NodeConfig(gpu_type, num_gpus, num_pcpus)

   resnet152 = load_resnet()
   alexnet = load_alexnet()

   if args.naive:
      stats = evaluate_sequential_naive(resnet152, alexnet, args.batch_size, args.trials, args.trial_length)
   elif args.fused:
      stats = evaluate_sequential_fused(resnet152, alexnet, args.batch_size, args.trials, args.trial_length)
   else:
      raise

   output = { 
               OUTPUT_KEY_STATS : stats, 
               OUTPUT_KEY_NODE_CONFIG : node_config.__dict__, 
               OUTPUT_KEY_TRIAL_LENGTH : args.trial_length,
               OUTPUT_KEY_NUM_TRIALS : args.trials
            }
   with open(args.output_path, "w") as f:
      json.dump(output, f, indent=4)

   print("Wrote results to file with path: {}".format(args.output_path))

