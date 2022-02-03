from wavelet import WT
from conf import device
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.transforms import ToTensor, Compose, Grayscale
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import beta as betadist

class MNIST:
  def __init__(self, batch_size=512):
    totensor = Compose([ToTensor(),lambda x: x.flatten()])
    self.name = "MNIST"
    self.train_data = datasets.MNIST('.', train=True, transform=totensor, target_transform=None, download=True)
    self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    self.test_data = datasets.MNIST('.', train=False, transform=totensor, target_transform=None, download=True)
    self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    self.train_loader.reset = lambda: None
    self.psi = lambda x: x
    self.psistar = lambda x : x
    self.train_data.reset = lambda: None

class CIFAR10Wavelet:
  def __init__(self, batch_size=256):

    self.name = 'CIFAR10Wavelet'
    self.wavelet = WT()
    totensor = Compose([ToTensor(), Grayscale(), lambda x: x.flatten()])

    def psi(x):
      return self.wavelet.wt(x.reshape(-1, 1, 32, 32), levels=2).reshape(-1, 1024)
    self.psi = psi
    def psistar(x):
      return self.wavelet.iwt(x.reshape(-1, 1, 32, 32), levels=2).reshape(-1, 1024)
    self.psistar = psistar
    self.train_data = datasets.CIFAR10('.', train=True, transform=totensor, target_transform=None, download=True)
    self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    self.test_data = datasets.CIFAR10('.', train=False, transform=totensor, target_transform=None, download=True)
    self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    self.train_loader.reset = lambda: None
    self.train_data.reset = lambda: None

class MNISTWavelet:
  def __init__(self, batch_size=512):
    self.name = 'MNISTWavelet'
    self.wavelet = WT()
    totensor = Compose([ToTensor(), lambda x: x.flatten()])
    def psi(x):
      return self.wavelet.wt(x.reshape(-1, 1, 28, 28), levels=1).reshape(-1, 784)
    self.psi = psi
    def psistar(x):
      return self.wavelet.iwt(x.reshape(-1, 1, 28, 28), levels=1).reshape(-1, 784)
    self.psistar = psistar
    self.train_data = datasets.MNIST('.', train=True, transform=totensor, target_transform=None, download=True)
    self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    self.test_data = datasets.MNIST('.', train=False, transform=totensor, target_transform=None, download=True)
    self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    self.train_loader.reset = lambda: None
    self.train_data.reset = lambda: None

class BoWDataset:
  def __init__(self, vec):
    self.vec = vec
  def __len__(self):
    return len(self.vec)
  def __getitem__(self, idx):
    return self.vec[idx], 0

class BagOfWords:
  def __init__(self, n, batch_size):
    twenty_train = fetch_20newsgroups(subset='train',
        shuffle=True, random_state=42)
    self.name = "BagOfWords"
    count_vect = CountVectorizer(max_features=n, max_df=0.2)
    count_vect.fit(twenty_train.data)
    self.train_data = torch.Tensor(count_vect.transform(twenty_train.data).todense().clip(0, 128))
    self.train_data.type(torch.float16)
    nor = self.train_data[self.train_data!=0].max()
    self.train_data = self.train_data / nor * 10

    del twenty_train
    twenty_test = fetch_20newsgroups(subset='test',
        shuffle=True, random_state=42)
    self.test_data = torch.Tensor(count_vect.transform(twenty_test.data).todense().clip(0, 128)) 
    self.test_data.type(torch.float16)
    self.test_data = self.test_data/ nor * 10
    del twenty_test

    self.train_loader = DataLoader(BoWDataset(self.train_data), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    self.test_loader = DataLoader(BoWDataset(self.test_data), batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
    self.train_loader.reset = lambda: None
    self.train_data.reset = lambda: None
    self.s = int((self.train_data!=0).float().sum(axis=1).mean())
    self.s_test = int((self.test_data!=0).float().sum(axis=1).mean())

    print("Average support size in train set:", self.s, " and in test set:", self.s_test)
    self.psi = lambda x:x
    self.psistar = lambda x:x


class BetaPriorSyntheticDataset(Dataset):
    """
    Dataset for creating sparse vectors.
    """

    def __init__(self, n, s, l, seed,
                 alpha=2, beta=8
                 ):
        self.n = n
        self.s = s
        self.l = l
        self.alpha = alpha
        self.beta = beta

        torch.manual_seed(seed)
        discretization = betadist.pdf(np.linspace(0.01, 0.99, self.n), alpha, beta)
        self.discretization_normalized = torch.tensor(discretization / np.sum(discretization))
        
        self.data = torch.zeros(l, self.n)

        self.reset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), torch.Tensor([0.0])

    def reset(self):
        self.i = 0
        index = torch.multinomial(#self.discretization_normalized.unsqueeze(0).repeat(self.l, 1)
            torch.zeros(self.l, self.n) + 1/self.n, self.s)
        sample = torch.tensor(betadist.rvs(self.alpha, self.beta, size=(self.s * self.l)), dtype=torch.float32).reshape(self.l, self.s)
        self.data = torch.zeros_like(self.data, memory_format=torch.legacy_contiguous_format).scatter_(1, index, sample)


class BernoulliSyntheticDataset(Dataset):
    """
    Dataset for creating sparse vectors.
    """
    def __init__(self, n, s, l, seed):
        self.n = n
        self.s = s
        self.l = l
        torch.manual_seed(seed)
        self.reset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), torch.Tensor([0.0])

    def reset(self):
        self.data = torch.zeros((self.l, self.n)) + self.s / self.n
        self.data = torch.bernoulli(self.data) * torch.normal(
            torch.zeros((self.l, self.n)), torch.ones((self.l, self.n))
        )

class Synthetic:
    """
    Synthetic dataset with train an test split.
    """

    def __init__(self, n, s_train, s_test, dataset, batch_size=512):
        self.n = n
        self.name = 'Synthetic'
        self.s = s_train
        self.train_data = dataset(n, s_train, 50000, seed=0)
        self.test_data = dataset(n, s_test, 10000, seed=1)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True, drop_last=True,
        )
        self.train_loader.reset = lambda: self.train_data.reset()

        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=batch_size, shuffle=False, drop_last=True,
        )
        self.psi = lambda x: x
        self.psistar = lambda x: x