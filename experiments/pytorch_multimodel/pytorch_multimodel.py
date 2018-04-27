import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time
import argparse

from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from torch.autograd import Variable
from datetime import datetime


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

def evaluate_resnet(resnet_model):
    evaluate_vision_model(resnet_model, "RESNET")

def evaluate_alexnet(alexnet_model):
    evaluate_vision_model(alexnet_model, "ALEXNET")

def evaluate_vision_model(model, name):
    preprocess = transforms.Compose(
            [transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
    )

    for _ in xrange(200):
        np_items = np.random.rand(50, 3, 224, 224)

        t1 = datetime.now()

        preprocessed_items = []
        for item in np_items:
            img_item = Image.fromarray(item, mode="RGB")
            img_item = preprocess(img_item)
            preprocessed_items.append(img_item)

        t2 = datetime.now()

        inp_var = Variable(torch.stack(preprocessed_items, dim=0))

        t3 = datetime.now()

        model(inp_var.cuda())

        t4 = datetime.now()

        print(name, (t4 - t3).total_seconds(), (t3 - t2).total_seconds(), (t2 - t1).total_seconds())

def evaluate_shared(resnet_model, alexnet_model):
   executor = ThreadPoolExecutor(max_workers=2)

   preprocess = transforms.Compose(
         [transforms.Scale(256),
         transforms.CenterCrop(224),
         transforms.ToTensor()]
   )

   for _ in xrange(200):
      np_items = np.random.rand(100, 3, 224, 224)

      t1 = datetime.now()

      resnet_preprocessed_items = []
      alexnet_preprocessed_items = []
      for i in range(len(np_items)):
         item = np_items[i]
         img_item = Image.fromarray(item, mode="RGB")
         img_item = preprocess(img_item)

         if i < 50:
            resnet_preprocessed_items.append(img_item)
         else:
            alexnet_preprocessed_items.append(img_item)

      t2 = datetime.now()

      resnet_var = Variable(torch.stack(resnet_preprocessed_items, dim=0))
      alexnet_var = Variable(torch.stack(alexnet_preprocessed_items, dim=0)) 

      t3 = datetime.now()

      f1 = executor.submit(resnet_model, resnet_var.cuda())
      f2 = executor.submit(alexnet_model, alexnet_var.cuda())

      f1.result()
      f2.result()

      t4 = datetime.now()

      print("SHARED", (t4 - t3).total_seconds(), (t3 - t2).total_seconds(), (t2 - t1).total_seconds())

def test_multithreaded():
    resnet152 = load_resnet()
    alexnet = load_alexnet()
    
    executor = ThreadPoolExecutor(max_workers=2)

    before = datetime.now() 
    f1 = executor.submit(evaluate_resnet, resnet152)
    time.sleep(2)
    f2 = executor.submit(evaluate_alexnet, alexnet)

    f1.result()
    f2.result()
    after = datetime.now() 

def test_multiprocess():
   from multiprocessing import Process

   def resnet_fn():
      os.system("taskset -p -c {} {}".format("0,16", os.getpid()))

      resnet152 = load_resnet()

      before = datetime.now()
      evaluate_resnet(resnet152)
      after = datetime.now()

      print((after - before).total_seconds())

   def alexnet_fn():
      os.system("taskset -p -c {} {}".format("1,17", os.getpid()))

      alexnet = load_alexnet()
      before = datetime.now()
      evaluate_alexnet(alexnet)
      after = datetime.now()

      print((after - before).total_seconds())

   p1 = Process(target=resnet_fn)
   p2 = Process(target=alexnet_fn)

   p1.start()
   p2.start()

   p1.join()
   p2.join()

def test_shared():
   resnet152 = load_resnet()
   alexnet = load_alexnet()

   evaluate_shared(resnet152, alexnet)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Pytorch multimodel experimentation")
   parser.add_argument('-p', '--multiproc', action='store_true', help="If specified, run in multiprocess mode")
   parser.add_argument('-t', '--multithread', action='store_true', help="If specified, run in multithread mode")
   parser.add_argument('-s', '--shared', action='store_true', help="If specified, run in shared preprocessing mode")

   args = parser.parse_args()

   if args.multiproc:
      test_multiprocess()
   elif args.multithread:
      test_multithreaded()
   else:
      test_shared()

   raise
