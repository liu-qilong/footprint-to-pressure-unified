import torch
import pandas as pd
from torch.utils.data import Dataset

from src.tool.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Pedar_Dataset_footprint2pressure(Dataset):
    def __init__(
            self: str,
            footprint_wrap_folder: str = 'data/processed/footprint-wrap',
            pedar_dynamic: str = 'data/processed/pedar_dynamic.pkl',
            sense_range: float = 600,
            dtype = torch.float32,
            ):
        self.pedar_dynamic = pd.read_pickle(pedar_dynamic)
        self.index = self.pedar_static.index
        self.dtype = dtype
        self.sense_range = sense_range
