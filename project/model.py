import torch
import torch.nn as nn


# Create the model
class MultiLayerPerceptron(nn.Module):
  def __init__(self, nin, nhidden, nout):
    super().__init__()
    self.main = nn.Sequential(
        # Flatten into simple feature vectors
        nn.Flatten(),
        nn.Linear(nin, nhidden),
        nn.ReLU(),
        nn.Linear(nhidden, nout)
        # Removed sigmoid
    )

  def forward(self, x):
    # Our model now returns logits!
    logits = self.main(x)
    return logits