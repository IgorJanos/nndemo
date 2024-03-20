import torch
import torch.nn.functional as tnf

import torchvision
import torchvision.transforms.functional as tvf


def image_transform(x):
  y = tvf.pil_to_tensor(x)  # Converts image into tensor (still uint8)
  y = (y / 255.0).float()   # Normalize into <0;1> range and float
  return y

def label_transform(x):
  y = torch.LongTensor([x])[0]                # Make tensor from integer
  y = tnf.one_hot(y, num_classes=10).float()  # Make one-hot vector from tensor
  return y


class DataModule:
  def __init__(self):

    self.dataset_train = torchvision.datasets.MNIST(
        root=".scratch/data/MNIST/train",
        train=True,                 # This is training set
        download=True,
        transform=image_transform,
        target_transform=label_transform
    )

    self.dataset_val = torchvision.datasets.MNIST(
        root=".scratch/data/MNIST/val",
        train=False,                # This is validation set
        download=True,
        transform=image_transform,
        target_transform=label_transform
    )

  def setup(self, cfg):
    self.dataloader_train = torch.utils.data.dataloader.DataLoader(
        self.dataset_train,
        batch_size=cfg.batch_size,    # Batch size hyper-parameter
        shuffle=True,                 # Iterate over samples in random order
        num_workers=cfg.num_workers   # Parallel processing of input samples
    )

    self.dataloader_val = torch.utils.data.dataloader.DataLoader(
        self.dataset_val,
        batch_size=cfg.batch_size,    # Batch size hyper-parameter
        shuffle=False,                # Do not use random ordering for validation
        num_workers=cfg.num_workers   # Parallel processing of input samples
    )

