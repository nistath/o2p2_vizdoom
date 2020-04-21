import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import math
from functools import lru_cache

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

    @lru_cache(maxsize=1)
    def get_all_idxs(self):
        return [tuple(p.name.split('_')[:2]) for p in self.png_dir.glob('*_screen.png')].sort()


class DoomSegmentedDataset(Dataset):
    def __init__(self, states, *args, **kwargs):
        self.states = np.load(states, allow_pickle=True)
        self.dataset = DoomSegmentationDataset(*args, **kwargs)

    @lru_cache(maxsize=1)
    def get_all_idxs(self):
        idxs = []
        for episode_key in self.states:
            episode = int(episode_key)
            for state in self.states[episode_key]:
                number = state['number'].item()
                for label in state['labels'].item():
                    idxs.append((episode, number, label['object_id']))

        return idxs

    def __getitem__(self, idx):
        episode, number, obj_id = idx

        screen, segmap = self.dataset[(episode, number)]
        return extract_segment(screen, segmap, obj_id)
