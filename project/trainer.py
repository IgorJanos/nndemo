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


    def setup(self, datamodule, log):
        # Setup data
        self.log = log
        self.datamodule = datamodule
        self.datamodule.setup(self.cfg)


    def fit(self):
        # Implement training loop
        self.log.on_training_start()
        for epoch in range(self.cfg.max_epochs):

            # Training phase
            stats_train = Statistics()
            self.train_epoch(
            epoch, 
            model=self.model, 
            dataloader=self.datamodule.dataloader_train, 
            stats=stats_train
            )

            # Validation phase
            stats_val = Statistics()
            self.validate_epoch(
            epoch, 
            model=self.model, 
            dataloader=self.datamodule.dataloader_val, 
            stats=stats_val
            )

            self.log.on_epoch_complete(
            epoch=epoch, 
            stats=Statistics.merge(stats_train, stats_val)
            )

        self.log.on_training_stop()



    def train_epoch(self, epoch, model, dataloader, stats):
        with tqdm(dataloader, desc=f"Train: {epoch}") as progress:
            for x,y in progress:

                # Move data to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                y_hat_logits = model(x)
                l = self.loss(y_hat_logits, y)

                # Backward pass & Update params
                self.opt.zero_grad()
                l.backward()
                self.opt.step()

                # Update statistics
                stats.step("loss_train", l.item())
                progress.set_postfix(stats.get())
       
       
    def validate_epoch(self, epoch, model, dataloader, stats):        
        with torch.no_grad():       # We don't need gradients in validation
            with tqdm(dataloader, desc=f"Val: {epoch}") as progress:
                for x,y in progress:
                    # Move data to GPU
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # Forward pass
                    y_hat_logits = model(x)
                    l = self.loss(y_hat_logits, y)

                    # Update statistics
                    stats.step("loss_val", l.item())
                    progress.set_postfix(stats.get())
