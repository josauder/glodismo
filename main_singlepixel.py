from data import MNIST, BernoulliSyntheticDataset, MNISTWavelet, Synthetic, BagOfWords
from models import NA_ALISTA, IHT
from train import run_experiment
from baseline import run_experiment_baseline, NeighborGenerator
from sensing_matrices import CompletelyLearned, SuperPixel, Pixel, LeftDRegularGraph, LoadedFromNumpy, ConstructedPooling
from noise import GaussianNoise, StudentNoise, Noiseless
import numpy as np
from conf import device
import matplotlib.pyplot as plt
import pandas as pd


def save_log(results, name):
  if len(results) == 2:
    train_logs, test_logs = results
  else:
    test_logs = results
    train_logs = False
  pd.DataFrame(test_logs).to_csv(name + "_test.csv", index=False)
  if train_logs:
    pd.DataFrame(train_logs).to_csv(name + "_train.csv", index=False)


n = 784
s = 50
from data import BernoulliSyntheticDataset
model = IHT(15, s)
epochs = 250

for data in [MNISTWavelet(), Synthetic(n, s, s, BernoulliSyntheticDataset, batch_size=512)]:
  for m in  [50, 200]:
    for seed in range(0, 10):
        losses = []
        scalars = []
        
        # Find optimal scaling factor!
        for scalar in np.linspace(0.1 , 1, 25):
          losses.append(run_experiment(
              n=n,
              sensing_matrix=Pixel(m, n, 32, scalar, seed, False), 
              model=model,
              data=data,
              use_mse=True,
              train_matrix=False,
              use_median=False,
              noise=GaussianNoise(40),
              epochs=0,
              positive_threshold=0.01,
              lr=0.0002,
          ))
          scalars.append(scalar)

        save_log(losses[np.argmin([x[1][0]["test_nmse"] for x in losses])], "results/singlepixel_random_" +data.name+"seed_"+str(seed)+"m_"+str(m))
        losses = [x[1][0]["test_nmse"] for x in losses]
        
        # Run Ours
        save_log(run_experiment(
            n=n,
            sensing_matrix=Pixel(m, n, 32, scalars[np.argmin(losses)]*0.9, seed, False), 
            model=model,
            data=data,
            use_mse=True,
            train_matrix=True,
            use_median=False,
            noise=GaussianNoise(40),
            epochs=epochs,
            positive_threshold=0.01,
            lr=0.0002,
            ), "results/singlepixel_learned_"+data.name+"seed_"+str(seed)+"m_"+str(m))

        # Run Simulated Annealing Baseline
        save_log(run_experiment_baseline(
            n=n,
            sensing_matrix=NeighborGenerator(Pixel(m, n, 32, scalars[np.argmin(losses)]*0.9, seed, False).to(device)), 
            model=model,
            data=data,
            use_mse=True,
            train_matrix=True,
            use_median=False,
            noise=GaussianNoise(40),
            epochs=epochs,
            positive_threshold=0.01,
            initial_temperature= 0.003,
            temperature_decay=0.9998,
            greedy=False,
        ),  "results/singlepixel_baseline_"+data.name+"seed_"+str(seed)+"m_"+str(m)+"_2")

        # Run Greedy Baseline
        save_log(run_experiment_baseline(
            n=n,
            sensing_matrix=NeighborGenerator(Pixel(m, n, 32, scalars[np.argmin(losses)]*0.95, seed, False).to(device)), 
            model=model,
            data=data,
            use_mse=True,
            train_matrix=True,
            use_median=False,
            noise=GaussianNoise(40),
            epochs=epochs,
            positive_threshold=0.01,
            initial_temperature= 0.0012,
            temperature_decay=0.99975,
            greedy=True,
        ), "results/singlepixel_baseline_greedy_"+data.name+"seed_"+str(seed)+"m_"+str(m)+"_2")
