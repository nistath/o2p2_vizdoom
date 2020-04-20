import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from PIL import Image
import numpy as np

from torchvision import transforms
from improc import extract_segment


class DoomSegmentationDataset(Dataset):
    def __init__(self, png_dir, desired_size=None):
        self.png_dir = png_dir = Path(png_dir)

        if desired_size is None:
            self.screen_tf = transforms.ToTensor()
            self.segmap_tf = self.screen_tf
        else:
            screen_tf = [
                transforms.Resize(desired_size, Image.BILINEAR),
                transforms.ToTensor(),
            ]
            segmap_tf = [
                transforms.Resize(desired_size, Image.NEAREST),
                np.array,
                torch.from_numpy,
            ]

            self.screen_tf = transforms.Compose(screen_tf)
            self.segmap_tf = transforms.Compose(segmap_tf)

    def __len__(self):
        return len(self.png_dir.glob('*_screen.png'))

    def __getitem__(self, idx):
        episode, number = idx

        def get_pil_img(name):
            return Image.open(self.png_dir.joinpath(f'{episode}_{number}_{name}.png'))

        screen = self.screen_tf(get_pil_img('screen'))
        segmap = self.segmap_tf(get_pil_img('labels'))

        return screen, segmap

    def get_all_idxs(self):
        return [tuple(p.name.split('_')[:2]) for p in self.png_dir.glob('*_screen.png')]


class DoomSegmentAutoencoderDataset(DoomSegmentationDataset):
    def __init__(self, png_dir):
        super().__init__(png_dir)

    def __getitem__(self, idx):
        raise NotImplementedError()
