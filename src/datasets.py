import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import math
from functools import lru_cache
from collections import defaultdict, Counter
import random
from copy import copy
import itertools

from improc import extract_segment
from torch.utils.data.sampler import Sampler


def shuffle(x, *args, **kwargs):
    random.shuffle(x, *args, **kwargs)
    return x


def shuffled_resample(x, length: int):
    if len(x) > length:
        return shuffle(x)[:length]

    if len(x) == length:
        return shuffle(x)

    shuffle(x)
    d, m = divmod(length, len(x))
    x = (x * d) + x[:m]
    return shuffle(x)


class SequentialSampler(list, Sampler):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)


class StratifiedRandomSampler(Sampler):
    def __init__(self, indices, label_fn, label_probability=None):
        self.labeled_idxs = defaultdict(list)
        self.label_fn = label_fn
        self.labels = set(map(label_fn, indices))

        if label_probability is None:
            P = 1 / len(self.labels)
            self.label_probability = {label: P for label in self.labels}
        else:
            self.label_probability = label_probability
            assert all(label in self.labels for label in self.label_probability)
        assert sum(self.label_probability.values()) == 1.0

        for idx in indices:
            label = label_fn(idx)
            if self.label_probability.get(label, 0) > 0:
                self.labeled_idxs[label].append(idx)

    def as_unshuffled_list(self):
        length = len(self)
        output = itertools.chain.from_iterable(shuffled_resample(copy(idxs),
                                                                 int(length * self.label_probability[label]))
                                               for label, idxs in self.labeled_idxs.items())
        return list(output)

    def as_list(self):
        return shuffled_resample(self.as_unshuffled_list(), length)

    def __iter__(self):
        return iter(self.as_list())

    def __len__(self):
        return sum(len(idxs) for idxs in self.labeled_idxs.values())


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
    def __init__(self, states, *args, blacklist=tuple(), **kwargs):
        self.states = np.load(states, allow_pickle=True)
        self.dataset = DoomSegmentationDataset(*args, **kwargs)
        self.blacklist = blacklist

    @lru_cache(maxsize=1)
    def get_all_idxs(self):
        idxs = []
        for episode_key in self.states:
            episode = int(episode_key)
            for state in self.states[episode_key]:
                number = state['number'].item()
                total = len(state['labels'].item()) + 1
                if total > 255:
                    raise ValueError('invalid number of labels')

                for i in range(total):
                    if i in self.blacklist:
                        continue

                    idxs.append((episode, number, i))

                # for label in state['labels'].item():
                #     idxs.append((episode, number, label['object_id']))

        return idxs

    def __getitem__(self, idx):
        episode, number, obj_id = idx

        screen, segmap = self.dataset[(episode, number)]
        # HACK: Fix object_id in states not correlating to buffer
        obj_id = torch.unique(segmap, sorted=True)[obj_id]

        return extract_segment(screen, segmap, obj_id)
