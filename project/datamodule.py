import torch
import torch.nn.functional as tnf

import torchvision
import torchvision.transforms as TF
import torchvision.transforms.functional as tvf


from project.dataset import create_animals_dataset, NumpyToTensor




class DataModule:
    def __init__(self):

        self.input_transform = TF.Compose([
            NumpyToTensor(),
            TF.Resize(size=(224, 224))
        ])

        self.dataset_train = create_animals_dataset(
            is_train=True,
            image_transform=self.input_transform
        )

        self.dataset_val = create_animals_dataset(
            is_train=False,
            image_transform=self.input_transform
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

