from argparse import Namespace


from project.model import MultiLayerPerceptron
from project.trainer import Trainer
from project.datamodule import DataModule


def main():

    cfg = Namespace(
    # Data module params
    batch_size = 16,
    num_workers = 2,

    # Training params
    max_epochs = 3,
    learning_rate = 0.1,

    # Model params
    num_hidden = 512
    )


    # Create model
    model = MultiLayerPerceptron(
    nin=28*28,                # Image size is 28x28
    nhidden=cfg.num_hidden,   # Larger hidden layer
    nout=10                   # 10 possible classes
    )

    # Create trainer & go go !
    trainer = Trainer(cfg, model)
    trainer.setup(DataModule())
    trainer.fit()

if __name__ == "__main__":
    main()
    

