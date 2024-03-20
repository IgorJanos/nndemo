import torch
import torch.nn as nn
from tqdm import tqdm


from project.utils import Statistics


def decide_device():
  if (torch.cuda.is_available()): return "cuda"
  #if (torch.backends.mps.is_available()): return "mps"
  return "cpu"



class Trainer:
  def __init__(self, cfg, model):

    # Select GPU device
    self.device = torch.device(decide_device())

    self.cfg = cfg
    self.model = model.to(self.device)      # Move parameters to GPU

    # Create optimizer
    self.opt = torch.optim.SGD(
        params=self.model.parameters(),
        lr=cfg.learning_rate
    )

    # Create loss function
    self.loss = nn.CrossEntropyLoss()


  def setup(self, datamodule):
    # Setup data
    self.datamodule = datamodule
    self.datamodule.setup(self.cfg)


  def fit(self):
    # Implement training loop
    for epoch in range(self.cfg.max_epochs):

        # Training phase
        stats = Statistics()
        with tqdm(
            self.datamodule.dataloader_train,
            desc=f"Train: {epoch} "
        ) as progress:
          for x,y in progress:

            # Move data to GPU
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            y_hat_logits = self.model(x)
            l = self.loss(y_hat_logits, y)

            # Backward pass & Update params
            self.opt.zero_grad()
            l.backward()
            self.opt.step()

            # Update statistics
            stats.step("loss_train", l.item())
            progress.set_postfix(stats.get())

        # Validation phase
        stats = Statistics()
        with torch.no_grad():       # We don't need gradients in validation
          with tqdm(
              self.datamodule.dataloader_val,
              desc=f"Val: {epoch} "
          ) as progress:
            for x,y in progress:

              # Move data to GPU
              x = x.to(self.device)
              y = y.to(self.device)

              # Forward pass
              y_hat_logits = self.model(x)
              l = self.loss(y_hat_logits, y)

              # Update statistics
              stats.step("loss_val", l.item())
              progress.set_postfix(stats.get())

