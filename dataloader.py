import torch
import torch.nn.functional as F
import csv
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

batches = []
DIR = './batch/'

def getValue(filename):
    f = open(filename, 'r')
    rdr = csv.reader(f)
    points=[]
    for line in rdr:
        if len(line):
            points_float = list(map(float, line)) # float로 변환
            points.append(points_float)
    # points = torch.tensor(points)
    batches.append(points)

def run():
    dir_list = os.listdir(DIR)
    for item in dir_list:
        getValue(DIR+str(item))

run()

class CustomDataset(Dataset):
    def __init__(self):
        self.data = batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(list(self.data[idx]))
        return x

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print(dataloader)
print('==Data==')
for idx,data in enumerate(dataloader):
    print(data)

# T=128
# initial value를 random point로 해서
# batch를 시작점으로 잡고 하기
# data size는 10000개 (initial point 10000개)
# T 는 128
# (x,y,z)는 3고정