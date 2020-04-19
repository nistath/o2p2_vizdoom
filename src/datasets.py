import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

from improc import extract_segment

pil2tensor = transforms.ToTensor()


class DoomSegmentationDataset(Dataset):
    def __init__(self, png_dir):
        self.png_dir = png_dir = Path(png_dir)

    def __len__(self):
        return len(self.png_dir.glob('*_screen.png'))

    def __getitem__(self, idx):
        episode, number = idx
        screen = pil2tensor(Image.open(
            self.png_dir.joinpath(f'{episode}_{number}_screen.png')))
        labels = torch.from_numpy(
            np.array(Image.open(self.png_dir.joinpath(f'{episode}_{number}_labels.png'))))

        return screen, labels

    def get_all_idxs(self):
        return [tuple(p.name.split('_')[:2]) for p in self.png_dir.glob('*_screen.png')]


class DoomSegmentAutoencoderDataset(DoomSegmentationDataset):
    def __init__(self, png_dir):
        super().__init__(png_dir)

    def __getitem__(self, idx):
        raise NotImplementedError()
