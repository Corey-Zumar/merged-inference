#!/bin/bash

while true;
do
  nvidia-smi --query-gpu="utilization.gpu" --format=csv >> $1
  sleep 1
done
