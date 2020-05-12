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
from itertools import chain
from typing import NamedTuple

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
    def __init__(self, indices, class_fn, label_probability=None):
        self.labeled_idxs = defaultdict(list)
        self.class_fn = class_fn
        self.labels = set(map(class_fn, indices))

        if label_probability is None:
            P = 1 / len(self.labels)
            self.label_probability = {label: P for label in self.labels}
        else:
            self.label_probability = label_probability
            assert all(label in self.labels for label in self.label_probability)
        assert 0.99 < sum(self.label_probability.values()) <= 1.0

        for idx in indices:
            label = class_fn(idx)
            if self.label_probability.get(label, 0) > 0:
                self.labeled_idxs[label].append(idx)

    def as_unshuffled_list(self, approx_len=None):
        length = approx_len or len(self)
        output = chain.from_iterable(shuffled_resample(copy(idxs),
                                                       int(length * self.label_probability[label]))
                                     for label, idxs in self.labeled_idxs.items())
        return list(output)

    def as_list(self, desired_len=None):
        length = desired_len or len(self)
        # HACK: There is a better way that doesn't sample with replacement here
        return shuffled_resample(self.as_unshuffled_list(length), length)

    def __iter__(self):
        return iter(self.as_list())

    def __len__(self):
        return sum(len(idxs) for idxs in self.labeled_idxs.values())


class FrameIdx(NamedTuple):
    episode: int
    frame: int


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

    def _get_pil_img(self, idx, name):
        episode, number = idx
        return Image.open(self.png_dir.joinpath(f'{episode}_{number}_{name}.png'))

    # @lru_cache(maxsize=20)
    def __getitem__(self, idx):
        screen = self.screen_tf(self._get_pil_img(idx, 'screen'))
        segmap = self.segmap_tf(self._get_pil_img(idx, 'labels'))

        return screen, segmap

    @lru_cache(maxsize=1)
    def get_all_idxs(self):
        idxs = [FrameIdx(p.name.split('_')[:2]) for p in self.png_dir.glob('*_screen.png')]
        idx.sort()
        return idxs


class ObjectIdx(NamedTuple):
    episode: int
    frame: int
    label_value: int
    object_name: str


class AlwaysContains:
    def __contains__(self, x):
        return True


class DoomSegmentedDataset(Dataset):
    def __init__(self, states, *args, blacklist=tuple(), **kwargs):
        self.states = np.load(states, allow_pickle=True)
        self.dataset = DoomSegmentationDataset(*args, **kwargs)
        self.blacklist = blacklist

    def get_all_idxs(self, whitelist=AlwaysContains()):
        idxs = []
        for episode_key in self.states:
            episode = int(episode_key)
            if episode_key not in whitelist:
                continue

            for state in self.states[episode_key]:
                number = state['number'].item()
                total = len(state['labels'].item()) + 1
                if total > 255:
                    raise ValueError('invalid number of labels')

                state_labels = ((x['value'], x['object_name'])
                                for x in state['labels'].item())
                labels = chain((
                    (0, 'Walls'),
                    (1, 'FloorCeil')),
                    state_labels)
                for label_id, name in labels:
                    if label_id not in self.blacklist:
                        idxs.append(ObjectIdx(episode, number, label_id, name))

        return idxs

    def get_episode_keys(self):
        return tuple(self.states.keys())

    def __getitem__(self, idx):
        if len(idx) == 3:
            episode, number, label_id = idx
        else:
            episode, number, label_id, _ = idx

        screen, segmap = self.dataset[(episode, number)]
        return extract_segment(screen, segmap, label_id)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

    def __getitem__(self, idx):
        return (idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


def idxs_in_same_frame(a, b):
    return a[:2] == b[:2]


def find_frame_boundary(idxs, start):
    for i in range(max(1, start), len(idxs)):
        if not idxs_in_same_frame(idxs[i - 1], idxs[i]):
            return i

    return len(idxs)


def idx_class(idx):
    return idx.object_name


def _corresponds(a, b):
    if a.label_value == b.label_value:
        return True

    if a.object_name == b.object_name:
        return True

    if a.object_name == 'Cacodemon':
        return b.object_name == 'DeadCacodemon'

    return False


def find_correspondences(*scenes):
    assert len(scenes) == 2

    correspondences = []
    others = set(scenes[1])
    for obj in scenes[0]:
        match = next((other for other in others if _corresponds(obj, other)), None)
        # TODO: ya might wanna assert we're not ambiguous here
        if match is not None:
            others.remove(match)
        correspondences.append((obj, match))

    return correspondences


class PredictionDataset(Dataset):
    def __init__(self, dataset: DoomSegmentedDataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

        scenes = []
        frame_start = 0
        while frame_start < len(idxs):
            frame_end = find_frame_boundary(idxs, frame_start + 1)
            scenes.append(idxs[frame_start:frame_end])
            frame_start = frame_end
        self.scenes = scenes

        # map from idx to idx
        correspondences = {}
        for i in range(1, len(scenes)):
            if scenes[i - 1][0].episode != scenes[i][0].episode:
                continue

            kv = find_correspondences(scenes[i - 1], scenes[i])
            correspondences.update((x for x in kv if x[1] is not None))
        self.correspondences = correspondences

        assert all(isinstance(x, ObjectIdx) for x in self.correspondences.keys())
        # TODO: Remove below if you want obj disappearance
        assert all(isinstance(x, ObjectIdx) for x in self.correspondences.values())

    def get_all_idxs(self):
        return self.correspondences.keys()

    def __getitem__(self, idx):
        start = self.dataset[idx]
        target = self.dataset[self.correspondences[idx]]
        return start, target
