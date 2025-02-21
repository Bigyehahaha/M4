import os

import h5py
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image
import os
import parser


class Datasets(Dataset):
    def __init__(self, args, fts_path:list, fts_label:list):
        self.features_path = fts_path
        self.features_label = fts_label

    def __len__(self):
        return len(self.features_path)

    def __getitem__(self, idx):
        slide_file = self.features_path[idx]
        label = self.features_label[idx]

        if slide_file.endswith('.pt'):
            features = torch.load(slide_file)

        return features, label

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]