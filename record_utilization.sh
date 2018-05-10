#!/bin/bash

while true;
do
  nvidia-smi --query-gpu="utilization.gpu" --format=csv >> out
  sleep 1
done
