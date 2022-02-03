from conf import device
import torch
from tqdm import tqdm
import numpy as np
from sensing_matrices import to_superpixel
from recovery import get_median_backward_op

def get_neighbor_rightdregular(phi):
    return get_neighbor_leftdregular(phi.T).T

def get_neighbor_leftdregular(phi):
    m, n = phi.shape
    random_col = np.random.randint(0, n)
    col = phi[:,random_col]
    col0 = (col == 0).int()
    col1 = 1 - col0
    where0 = torch.where(col0)[0]
    
    where1 = torch.where(col1)[0]
    i = where0[np.random.randint(len(where0))]
    j = where1[np.random.randint(len(where1))]
    phi[i, random_col] = phi.max()
    phi[j, random_col] = 0
    return phi

class NeighborGenerator:

  def __init__(self, sensing_matrix, left=False, superpixel=False):
    self.phi = sensing_matrix(1, test=True)[0]
    self.d = sensing_matrix.d
    self.left = left
    self.superpixel = superpixel

  def __call__(self, c, test=False):
    if test:
      return self.phi.unsqueeze(0).repeat(c, 1, 1)

    if not self.left:
        phi_ = get_neighbor_rightdregular(torch.clone(self.phi))
    else:
        phi_ = get_neighbor_leftdregular(torch.clone(self.phi))
    if self.superpixel:
        phi_ = to_superpixel(phi_)

    return phi_.to(device).unsqueeze(0).repeat(c, 1, 1)

  def to(self, device):
    print("Not implemented")
    return self

def test_epoch_(model, sensing_matrix, data, noise, use_median, n, positive_threshold):
  with torch.no_grad():
    test_loss_l2 = 0
    test_normalizer_l2 = 0
    test_loss_l1 = 0
    test_normalizer_l1 = 0

    phi = sensing_matrix(1, test=True)
    phi = phi[0]    
    forward_op = lambda x: torch.matmul(x, phi.T)
    backward_op = lambda x: torch.matmul(x, phi)
    false_positives = []
    false_negatives = []
    
    if use_median:
      backward_op = get_median_backward_op(phi, n, sensing_matrix.d, test=True)

  
    for iteration, (X, _) in tqdm(enumerate(iter(data.test_loader))):
      X = X.to(device) 
      y = noise(forward_op(X))

      Xhat = model(y, forward_op, backward_op, data.psi, data.psistar)
      Xhat = data.psistar(Xhat)
      loss = ((Xhat-X)**2).mean()
      test_normalizer_l2 += (X ** 2).mean().item()
      test_loss_l2 += loss.item()
      test_normalizer_l1 += (torch.abs(X)).mean().item()
      test_loss_l1 += torch.abs(Xhat - X).mean().item()
      true_positives = (torch.abs(X) >= positive_threshold).int()
      detected_positives = (torch.abs(Xhat) >= positive_threshold).int()

      false_positives.append((detected_positives * (1 - true_positives)).float().mean().item())
      false_negatives.append((true_positives * (1 - detected_positives)).float().mean().item())
    return {
      "test_loss_l2": test_loss_l2,
      "test_loss_l1": test_loss_l1,
      "test_nmse": 10 * np.log10(test_loss_l2 / test_normalizer_l2),
      "test_nmae": 10 * np.log10(test_loss_l1 / test_normalizer_l1),
      "test_false_positives": np.mean(false_positives),
      "test_false_negatives": np.mean(false_negatives),
    }

def train_epoch_(model, sensing_matrix, data, noise, use_median, n, positive_threshold, opt, use_mse):
  train_loss_l2 = 0
  train_normalizer_l2 = 0
  train_loss_l1 = 0
  train_normalizer_l1 = 0
  false_positives = []
  false_negatives = []
  with torch.no_grad():
    accept = []
    for iteration, (X, _) in tqdm(enumerate(iter(data.train_loader))):
      X = X.to(device)

      opt.zero_grad()
      phi = sensing_matrix(X.shape[0])

      forward_op = lambda x: torch.bmm(x.unsqueeze(1), phi.transpose(1, 2)).squeeze(1)
      if not use_median:
        backward_op = lambda x: torch.bmm(x.unsqueeze(1), phi).squeeze(1)
      else:
        backward_op = get_median_backward_op(phi, n, sensing_matrix.d, test=False, train_matrix=True)

      y = noise(forward_op(X))
      Xhatt = model(y, forward_op, backward_op, data.psi, data.psistar)
      Xhat = data.psistar(Xhatt)

      true_positives = (torch.abs(X) >= positive_threshold).int()
      detected_positives = (torch.abs(Xhat) >= positive_threshold).int()

      if use_mse:
        loss = ((Xhat-X)**2).mean()
      else:
        loss = (torch.abs(Xhat-X)).mean()

      if opt.step(loss):
        accept.append(1)
        sensing_matrix.phi = phi[0]
      else:
        accept.append(0)

      train_normalizer_l2 += (X ** 2).mean().item()
      train_loss_l2 += ((Xhat-X)**2).mean().item()
      train_normalizer_l1 += (torch.abs(X)).mean().item()
      train_loss_l1 += torch.abs(Xhat - X).mean().item()

      false_positives.append((detected_positives * (1 - true_positives)).float().mean().item())
      false_negatives.append((true_positives * (1 - detected_positives)).float().mean().item())
    return {
      "train_loss_l2": train_loss_l2,
      "train_loss_l1": train_loss_l1,
      "train_nmse": 10 * np.log10(train_loss_l2 / train_normalizer_l2),
      "train_nmae": 10 * np.log10(train_loss_l1 / train_normalizer_l1),
      "train_false_positives": np.mean(false_positives),
      "train_false_negatives": np.mean(false_negatives),
      "train_acceptance_rate": np.mean(accept),
    }

def run_experiment_baseline(
    n,
    sensing_matrix, 
    model,
    data,
    use_mse,
    train_matrix,
    use_median,
    noise,
    epochs,
    positive_threshold,
    initial_temperature,
    temperature_decay,
    greedy,
    test_model=False,
    ):

  model = model.to(device)
  sensing_matrix = sensing_matrix.to(device)
  
  train_metrics = []
  if not test_model:
    test_model = model 
 
  test_metrics = [test_epoch_(test_model, sensing_matrix, data, noise, use_median, n, positive_threshold)]
  print("Epoch: 0 Test NMSE:", test_metrics[-1]['test_nmse'],  "Test NMAE:", test_metrics[-1]['test_nmae'])

  "Only train if algorithm or matrix are learnable"
  if (train_matrix):  
    opt = BaselineOptimizer(initial_temperature, temperature_decay, greedy=greedy)

    for epoch in range(epochs):
      metrics = train_epoch_(model, sensing_matrix, data, noise, use_median, n, positive_threshold, opt, use_mse, train_matrix)
      train_metrics.append(metrics)
      data.train_data.reset()
      test_metrics.append(test_epoch_(test_model, sensing_matrix, data, noise, use_median, n, positive_threshold))
      print("Epoch:", epoch+1, "Test NMSE:", test_metrics[-1]['test_nmse'],  "Test NMAE:", test_metrics[-1]['test_nmae'], "Train NMSE:", train_metrics[-1]['train_nmse'],  "Train NMAE:", train_metrics[-1]['train_nmae'])
  return train_metrics, test_metrics

class BaselineOptimizer:

    def __init__(self, initial_temperature, temperature_decay, greedy):
        """Simulated Annealing & Greedy Baselines
        If greedy, temperarture params are ignored"""
        self.prev_loss = torch.tensor(10000)
        self.temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.greedy = greedy

    def zero_grad(self):
        pass

    def step(self, loss):
        accept = False
        if loss < self.prev_loss:
           accept = True
        else:
          if not self.greedy:
            val = torch.exp((self.prev_loss - loss).cpu() / self.temperature)
            if val > torch.rand(1):
              accept = True
            self.temperature = self.temperature * self.temperature_decay
        if accept:
            self.prev_loss = loss
        return accept
      
    