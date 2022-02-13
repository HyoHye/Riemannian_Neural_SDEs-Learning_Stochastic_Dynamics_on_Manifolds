import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


def load_data(data_path, batch_size=1, nb_workers=0):
    """
    Loads data from CustomDataset.
    Args:
        - data_path: path to dataset
        - batch_size: train and test batch size
        - nb_workers: number of workers for dataloader
    """

    data = torch.tensor(np.load(data_path))
    dataset = CustomDataset(data)

    train_len = int(len(dataset)*TRAIN_RATIO)
    validation_len = int(len(dataset)*VALIDATION_RATIO)

    train_dataset = dataset[:train_len]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = nb_workers)

    validation_dataset = dataset[train_len:train_len+validation_len]
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers = nb_workers)

    test_dataset = dataset[train_len+validation_len:]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = nb_workers)


    return train_loader, validation_loader, test_loader