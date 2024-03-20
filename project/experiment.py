
from pathlib import Path
import yaml
import torch

from torchsummary import summary
from project.model import MultiLayerPerceptron
from project.datamodule import DataModule
from project.trainer import Trainer
import project.logging as L

BASE_PATH=Path(".scratch/experiments")


class Experiment:
    def __init__(
        self,
        cfg            
    ):
        self.cfg = cfg
        self.experiment_path = BASE_PATH / cfg.name / cfg.ver

        # Print some info
        print(f" > Created experiment : {cfg.name}/{cfg.ver}")
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self._save_config(cfg, self.experiment_path)

        # Create model & datamodule
        self.model = self._create_model(cfg)
        self.datamodule = DataModule()

        # Create trainer
        self.trainer = Trainer(cfg, self.model)

    def _create_model(self, cfg):
        model = MultiLayerPerceptron(
            nin=28*28,                # Image size is 28x28
            nhidden=cfg.num_hidden,   # Larger hidden layer
            nout=10                   # 10 possible classes
        )

        print(" Creating model: ")
        summary(
            model,
            input_size=(1,28,28),
            batch_size=1,
            device="cpu"
        )
        return model

    def _save_config(self, cfg, exp_path):
        config_yaml = yaml.dump(vars(cfg))

        print(" > Training Configuration:")
        print("---------------------------")
        print(config_yaml.strip())
        print("\n")

        # Save configuration into YAML
        (exp_path / "config.yaml").write_text(config_yaml)



    def train(self):

        # Setup training
        self.trainer.setup(
            datamodule=self.datamodule,
            logs=[
                L.CSVLog(self.experiment_path / "training.csv"),
                L.ReportCompiler(
                    filepath=self.experiment_path / "report.pdf",
                    source_filepath=self.experiment_path / "training.csv"
                ),
                L.ModelCheckpointer(self)
            ]
        )
        self.trainer.fit()



    def save_checkpoint(self, filename):

        # Both - model and optimizer
        checkpoint = {
            "model": self.model.state_dict(),
            "opt": self.trainer.opt.state_dict()
        }

        # Create checkpoints folder
        file_path = self.experiment_path / "checkpoints" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint: {file_path.as_posix()}")
        torch.save(checkpoint, file_path.as_posix())
