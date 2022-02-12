import numpy as np
import torch
from dataloader import CustomDataset

from torch.utils.data import DataLoader

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

data = torch.tensor(np.load('without_noise.npy'))
dataset = CustomDataset(data)

train_len = int(len(dataset)*TRAIN_RATIO)
validation_len = int(len(dataset)*VALIDATION_RATIO)

train_dataset = dataset[:train_len]
validation_dataset = dataset[train_len:train_len+validation_len]
test_dataset = dataset[train_len+validation_len:]

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)