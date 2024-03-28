from argparse import Namespace, ArgumentParser


from project.experiment import Experiment


def main(cfg):

    # Start a new training experiment
    experiment = Experiment(cfg)
    experiment.train()


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("--name", "-n", type=str, default="conv", help="Experiment name")
    p.add_argument("--ver", "-v", type=str, default="debug", help="Experiment version")

    p.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size")
    p.add_argument("--num_workers", "-nw", type=int, default=2, help="Number of Dataloader workers")
    p.add_argument("--max_epochs", "-e", type=int, default=3, help="Number of epochs to train")
    p.add_argument("--learning_rate", "-lr", type=float, default=0.1, help="Optimizer learning rate")
    p.add_argument("--num_hidden", "-nh", type=int, default=512, help="Number of hidden units")

    # Model architecture
    p.add_argument("--model_architecture", "-ma",
                   choices=[
                       "simple", "resnet", "vit"
                   ],
                   default="simple",
                   help="Model architecture"
                   )

    cfg = p.parse_args()
    main(cfg)


