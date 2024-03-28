from typing import Callable
import cv2
from pathlib import Path
import urllib.request
import subprocess
import numpy as np
import torch
import torch.nn.functional as tnf
from torch.utils.data import Dataset

DATASET_BASEPATH=Path(".scratch/data")
ANIMALS_DOWNLOAD_URL="https://vggnas.fiit.stuba.sk/download/datasets/animals/animals.zip"


class NumpyToTensor(Callable):
    def __call__(self, x):
        x = np.transpose(x, axes=(2,0,1))       # HWC -> CHW
        x = torch.from_numpy(x) / 255.0         # <0;255>UINT8 -> <0;1>
        return x.float()                        # cast as 32-bit flow


class ClassToOneHot(Callable):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        y = torch.LongTensor([x])[0]
        y = tnf.one_hot(y, num_classes=self.num_classes)
        return y.float()



class AnimalsDataset(Dataset):
    def __init__(
        self,
        subset_path: Path,
        image_transform
    ):
        
        self.class_names = {}
        self.files = sorted(list(subset_path.rglob("*.jpg")))
        self.image_transform = image_transform

        # Assemble class dictionary
        for filepath in self.files:
            class_name = filepath.parent.name
            if not class_name in self.class_names:
                self.class_names[class_name] = len(self.class_names)

        # Total number of classes
        self.num_classes = len(self.class_names)
        self.label_transform = ClassToOneHot(self.num_classes)


    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, index):
        # Get what we need
        file_path = self.files[index]
        class_name = file_path.parent.name
        label_class = self.class_names[class_name]

        # Load image
        image = self._load_image(file_path)
        if (self.image_transform is not None):
            image = self.image_transform(image)

        # Return what we've got
        label = self.label_transform(label_class)
        return image, label


    def _load_image(self, file_path):
        image = cv2.imread(file_path.as_posix(), cv2.IMREAD_COLOR)      # <H;W;C>
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image





def download_file(url, local_file):
    try:
        print("Downloading file: ", url)
        urllib.request.urlretrieve(url, local_file)
        return True
    except:
        print("Error downloading: ", url)
        return False
    

def extract_archive(file_path):
    print(f"Extracting {file_path.as_posix()}...")
    if (file_path.suffix == ".zip"):
        target_dir = file_path.parent
        subprocess.call([
            "unzip", "-q", "-o", file_path.as_posix(),
            "-d", target_dir
        ])


def create_animals_dataset(is_train=True, **kwargs):

    # Download if necessary
    local_dataset_folder = DATASET_BASEPATH / "animals"
    local_dataset_file = DATASET_BASEPATH / "animals.zip"
    check_file = local_dataset_folder / ".done"
    if (not check_file.exists()):
        ok = download_file(ANIMALS_DOWNLOAD_URL, local_dataset_file.as_posix())
        if (not ok):
            return None
        
        # Unzip
        extract_archive(local_dataset_file)
        # Touch file - we are done!
        check_file.touch()

    # Create dataset from local folder
    if (is_train):
        images_folder = local_dataset_folder / "train"
    else:
        images_folder = local_dataset_folder / "val"

    # Return the proper dataset instance
    return AnimalsDataset(images_folder, **kwargs)



    
