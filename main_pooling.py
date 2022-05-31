from data import MNIST, BernoulliSyntheticDataset, MNISTWavelet, Synthetic, BagOfWords
from recovery import NA_ALISTA, IHT, NNLAD
from baseline import run_experiment_baseline, NeighborGenerator
from sensing_matrices import Pooling, ConstructedPooling
from noise import GaussianNoise, StudentNoise, Noiseless
import numpy as np
from conf import device
import matplotlib.pyplot as plt
from train import run_experiment
import pandas as pd
import torch.nn.functional as F
from conf import device
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn


def save_log(results, name):
  if len(results) == 2:
    train_logs, test_logs = results
  else:
    test_logs = results
    train_logs = False
  pd.DataFrame(test_logs).to_csv(name + "_test.csv", index=False)
  if train_logs:
    pd.DataFrame(train_logs).to_csv(name + "_train.csv", index=False)


n = 961
s = 80
m = 248
from data import BetaPriorSyntheticDataset
model = NNLAD(200,0.1, 0.6)
test_model = NNLAD(1000, 0.1, 0.6)
epochs = 50

for data in [Synthetic(n, s, s, BetaPriorSyntheticDataset, batch_size=512)]:

  for seed in range(10):
      losses = []
      scalars = []
      print("Determining optimal scaling factor")
      for scalar in np.linspace(0.9, 1.05, 3):
        print(scalar)
        losses.append(run_experiment(
            n=n,
            sensing_matrix=Pooling(m, n, 31, scalar, seed), 
            model=model,
            data=data,
            use_mse=False,
            train_matrix=False,
            use_median=False,
            noise=GaussianNoise(40),
            epochs=0,
            positive_threshold=0.01,
            lr=0.0002,
            test_model=test_model,
        ))
        scalars.append(scalar)

      save_log(losses[np.argmin([x[1][0]["test_nmae"] for x in losses])], "results/pooling_random_" +data.name+"seed_"+str(seed))
      losses = [x[1][0]["test_nmae"] for x in losses]

      print("Training Pooling Matrix using GLODISMO")
      save_log(run_experiment(
          n=n,
          sensing_matrix=Pooling(m, n, 31, scalars[np.argmin(losses)], seed), 
          model=model,
          data=data,
          use_mse=False,
          train_matrix=True,
          use_median=False,
          noise=GaussianNoise(40),
          epochs=epochs,
          positive_threshold=0.01,
          lr=0.00002,
          test_model=test_model,
          use_greedy_stabilization=False,
      ), "results/pooling_learned_"+data.name+"seed_"+str(seed))
      
      print("Simulated Annealing Baseline")
      save_log(run_experiment_baseline(
          n=n,
          sensing_matrix=NeighborGenerator(Pooling(m, n, 31, scalars[np.argmin(losses)], seed).to(device)), 
          model=model,
          data=data,
          use_mse=False,
          train_matrix=True,
          use_median=False,
          noise=GaussianNoise(40),
          epochs=epochs,
          positive_threshold=0.01,
          initial_temperature= 0.0006,
          temperature_decay=0.9997,
          greedy=False,
          test_model=test_model,
      ),  "results/pooling_baseline_"+data.name+"seed_"+str(seed))
      
      print("Greedy Baseline")
      save_log(run_experiment_baseline(
          n=n,
          sensing_matrix=NeighborGenerator(Pooling(m, n, 31, scalars[np.argmin(losses)], seed).to(device)), 
          model=model,
          data=data,
          use_mse=False,
          train_matrix=True,
          use_median=False,
          noise=GaussianNoise(40),
          epochs=epochs,
          positive_threshold=0.01,
          initial_temperature= 0.0006,
          temperature_decay=0.9997,
          greedy=True,
          test_model=test_model,
      ), "results/pooling_baseline_greedy_"+data.name+"seed_"+str(seed))
      

