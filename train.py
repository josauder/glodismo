import torch.nn.functional as F
from conf import device
import torch
from tqdm import tqdm
import numpy as np
from recovery import get_median_backward_op

def test_epoch(model, sensing_matrix, data, noise, use_median, n, positive_threshold):
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

def train_epoch(model, sensing_matrix, data, noise, use_median, n, positive_threshold, opt, use_mse, train_matrix):
  train_loss_l2 = 0
  train_normalizer_l2 = 0
  train_loss_l1 = 0
  train_normalizer_l1 = 0
  false_positives = []
  false_negatives = []
  
  for iteration, (X, _) in tqdm(enumerate(iter(data.train_loader))):
    X = X.to(device)

    opt.zero_grad()
    if train_matrix:
      phi = sensing_matrix(X.shape[0])
      forward_op = lambda x: torch.bmm(x.unsqueeze(1), phi.transpose(1, 2)).squeeze(1)
      if not use_median:
        backward_op = lambda x: torch.bmm(x.unsqueeze(1), phi).squeeze(1)
      else:
        backward_op = get_median_backward_op(phi, n, sensing_matrix.d, test=False, train_matrix=True)
    else:
      phi = sensing_matrix(1, test=True)
      forward_op = lambda x: torch.matmul(x, phi[0].T)
      if not use_median:    
        backward_op = lambda x: torch.matmul(x, phi[0])
      else:
        backward_op = get_median_backward_op(phi, n, sensing_matrix.d, test=False, train_matrix=False)

      

    y = noise(forward_op(X))
    Xhatt = model(y, forward_op, backward_op, data.psi, data.psistar)
    Xhat = data.psistar(Xhatt)

    true_positives = (torch.abs(X) >= positive_threshold).int()
    detected_positives = (torch.abs(Xhat) >= positive_threshold).int()

    if use_mse:
      loss = ((Xhat-X)**2).mean()
    else:
      loss = (torch.abs(Xhat-X)).mean()

    loss.backward()
    opt.step()

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
  }

def run_experiment(
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
    lr,
    test_model=False,
    ):

  model = model.to(device)
  if not test_model:
    test_model = model 
  sensing_matrix = sensing_matrix.to(device)
  
  train_metrics = []
  test_metrics = [test_epoch(test_model, sensing_matrix, data, noise, use_median, n, positive_threshold)]
  print("Epoch: 0 Test NMSE:", test_metrics[-1]['test_nmse'],   "Test NMAE:", test_metrics[-1]['test_nmae'])

  "Only train if algorithm or matrix are learnable"
  if (len(list(model.parameters()))>0 or train_matrix):

    learnable_params = list(model.parameters())
    if train_matrix:
      print('Training Matrix!')
      learnable_params.extend(list(sensing_matrix.parameters()))
    opt = torch.optim.Adam(learnable_params, lr=lr)

    for epoch in range(epochs):
      train_metrics.append(train_epoch(model, sensing_matrix, data, noise, use_median, n, positive_threshold, opt, use_mse, train_matrix))
      data.train_data.reset()
      test_metrics.append(test_epoch(test_model, sensing_matrix, data, noise, use_median, n, positive_threshold))
      print("Epoch:", epoch+1, "Test NMSE:", test_metrics[-1]['test_nmse'],   "Test NMAE:", test_metrics[-1]['test_nmae'], "Train NMSE:", train_metrics[-1]['train_nmse'],  "Train NMAE:", train_metrics[-1]['train_nmae'])

  return train_metrics, test_metrics
