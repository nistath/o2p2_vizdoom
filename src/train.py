import torch
import torch.cuda
import torchvision.models as models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import shutil
from pathlib import Path

from datasets import DoomSegmentationDataset, DoomSegmentedDataset, SequentialSampler
from models.loss import LossNetwork
from improc import *
from models.perception import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f'Using device {device}.')

    val_path = Path('/home/nistath/Desktop/val')

    img_size = (240, 320)
    dataset = DoomSegmentedDataset('/home/nistath/Desktop/run1/states.npz',
                                   '/home/nistath/Desktop/run1/images/', desired_size=img_size)

    num_features = 256

    split = 0.9
    batch_size = 32

    model = torch.nn.Sequential(
        Perception((3,) + img_size, num_features),
        InversePerceptionConv((3,) + img_size, num_features),
    ).to(device)

    if True:
        all_idxs = dataset.get_all_idxs()
        random.shuffle(all_idxs)
        split_point = int(split * len(all_idxs))
        trn_idxs = all_idxs[:split_point]
        val_idxs = all_idxs[split_point:]
        del all_idxs
        trn_sampler = SubsetRandomSampler(trn_idxs)
        trn_dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=4, sampler=trn_sampler)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # opt = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9)
        MSELoss = torch.nn.MSELoss()

        # vgg_model = models.vgg16(pretrained=True).to(device)
        # loss_network = LossNetwork(vgg_model)
        # loss_network.eval()

        model.train()
        print('Starting training.')
        for epoch in trange(2):
            for imgs in tqdm(trn_dataloader):
                imgs = imgs.to(device)

                opt.zero_grad()
                imgs_hat = model(imgs)
                # print(imgs.shape, imgs_hat.shape); exit()
                loss = MSELoss(imgs, imgs_hat)
                del imgs
                del imgs_hat
                loss.backward()
                opt.step()
                tqdm.write(f'Loss: {loss.data.item()}')

        torch.save(model.state_dict(), 'model.pth')
        torch.save(trn_idxs, 'trn_idxs.pth')
        torch.save(val_idxs, 'val_idxs.pth')
        val_sampler = SubsetRandomSampler(val_idxs)
    else:
        model.load_state_dict(torch.load('model.pth'))
        trn_idxs = torch.load('trn_idxs.pth')
        val_idxs = torch.load('val_idxs.pth')

    val_sampler = SequentialSampler(trn_idxs[:30])
    # val_sampler = SubsetRandomSampler(val_idxs)
    val_dataloader = DataLoader(
        dataset, batch_size=8, num_workers=0, sampler=val_sampler)

    if val_path.exists():
        shutil.rmtree(val_path)
    val_path.mkdir()

    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(tqdm(val_dataloader)):
            imgs = imgs.to(device)

            imgs_hat = model(imgs)

            grid_img = make_grid(
                torch.cat((imgs.cpu(), imgs_hat.cpu())), nrow=imgs.shape[0])
            tensor2pil(grid_img).save(val_path.joinpath(f'{i}.png'))
