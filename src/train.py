import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

from improc import extract_segment, torch_to_normal_shape
from datasets import DoomSegmentationDataset
from models.perception import Perception, InversePerception

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f'Using device {device}.')

    dataset = DoomSegmentationDataset('/home/nistath/Desktop/run1/images/')

    split = 0.05
    batch_size = 1

    all_idxs = dataset.get_all_idxs()
    random.shuffle(all_idxs)
    split_point = int(split * len(all_idxs))
    trn_sampler = SubsetRandomSampler(all_idxs[:split_point])
    val_sampler = SubsetRandomSampler(all_idxs[split_point:])
    trn_dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, sampler=trn_sampler)
    val_dataloader = DataLoader(
        dataset, batch_size=1, num_workers=0, sampler=val_sampler)

    model = torch.nn.Sequential(
        Perception((3, 480, 640), 256),
        InversePerception((3, 480, 640), 256),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    if False:
        model.train()
        print('Starting training.')
        for screens, segmaps in tqdm(trn_dataloader):
            imgs = []
            for screen, segmap in zip(screens, segmaps):
                # TODO: Get unique from the npz
                for obj_id in torch.unique(segmap):
                    img = extract_segment(screen, segmap, obj_id)
                    imgs.append(img)

            imgs = torch.stack(imgs).to(device)
            opt.zero_grad()
            imgs_hat = model(imgs)
            loss = criterion(imgs, imgs_hat)
            loss.backward()
            opt.step()
            print(loss.data.item())

        torch.save(model.state_dict(), 'model.pth')
    else:
        model.load_state_dict(torch.load('model.pth'))

    model.eval()
    with torch.no_grad():
        for screens, segmaps in tqdm(val_dataloader):
            imgs = []
            for screen, segmap in zip(screens, segmaps):
                # TODO: Get unique from the npz
                for obj_id in torch.unique(segmap):
                    img = extract_segment(screen, segmap, obj_id)
                    imgs.append(img)

            imgs = torch.stack(imgs).to(device)
            imgs_hat = model(imgs)
            f, ax = plt.subplots(2, imgs.shape[0])
            for i in range(imgs.shape[0]):
                ax[0, i].imshow(torch_to_normal_shape(imgs[i]))
                ax[1, i].imshow(torch_to_normal_shape(imgs_hat[i]))
            plt.show()
