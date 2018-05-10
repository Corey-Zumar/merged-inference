#!/usr/bin/env bash
for i in 2 4 8 16 32 64 128 256 512
do   
  python mnist_dense_ensemble.py --n_ensemble=8 --epochs=0.2 --serving_model_path=models/dense_experimental/ --combine_dense --n_trials=200 --batch_size=$i
done
for i in 2 4 8 16 32 64 128 256 512
do
  python mnist_dense_ensemble.py --n_ensemble=8 --epochs=0.2 --serving_model_path=models/dense_experimental/ --n_trials=200 --batch_size=$i
done
