import torch
import torch.cuda
import torchvision.models as models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import shutil
from pathlib import Path
from datetime import datetime

from datasets import *
from improc import *
from models.loss import masked_mse_loss
from models.perception import *
from models.prediction import *
from models.inspect import Inspect
from PerceptualSimilarity.models import PerceptualLoss


def cutoff_idxs(idxs, max_len):
    if max_len is None:
        return idxs
    return idxs[:find_frame_boundary(idxs, max_len)]


if __name__ == '__main__':
    torch.manual_seed(6969)
    random.seed(6969)
    np.random.seed(6969)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f'Using device {device}.')

    experiment_name = datetime.now().isoformat()
    experiment_name += '_medium_filtersok_preddict'
    results_path = Path('/home/nistath/Desktop/val/')
    val_path = results_path.joinpath(experiment_name)
    load_path = results_path.joinpath('2020-05-12T00:23:10.810421_perceptual/save')
    save_path = val_path.joinpath('save')

    img_shape = (240, 320)
    dataset = DoomSegmentedDataset('/home/nistath/Desktop/run2/states.npz',
                                   '/home/nistath/Desktop/run2/images/', desired_size=img_shape,
                                   #    blacklist=(0, 1,)
                                   )

    mask_mse_loss = True
    use_stratification = True
    use_perceptual_loss = True
    reuse_split = True
    reuse_autoencoder = True  # implies split will be reused
    validate_autoencoder = False

    if reuse_autoencoder:
        if val_path.exists():
            raise ValueError('will not overwrite')
    else:
        save_path.mkdir(exist_ok=True, parents=True)
    val_path.mkdir(exist_ok=True, parents=True)

    cheat = False

    split = 0.5
    max_len_trn = 3000
    max_len_val = None
    max_val_num = 1000
    batch_size = 32

    # num_features = 256
    # enc = Perception((3,) + img_shape, num_features)
    # dec = InversePerception((3,) + img_shape, num_features)
    enc, dec = ConvAutoencoder((3,) + img_shape, have_linear=False)
    model = torch.nn.Sequential(enc, dec).to(device)

    if not reuse_autoencoder:
        if reuse_split:
            trn_idxs = torch.load(load_path.joinpath('trn_idxs.pth'))
            val_idxs = torch.load(load_path.joinpath('val_idxs.pth'))
        else:
            if False:
                all_idxs = dataset.get_all_idxs()
                random.shuffle(all_idxs)
                split_point = find_frame_boundary(
                    all_idxs, int(split * len(all_idxs)))
                trn_idxs = all_idxs[:split_point]
                val_idxs = all_idxs[split_point:]
                del all_idxs
            else:
                episode_keys = shuffle(list(dataset.get_episode_keys()))
                split_point = int(split * len(episode_keys))
                trn_idxs = cutoff_idxs(dataset.get_all_idxs(
                    shuffle(episode_keys[:split_point])), max_len_trn)
                val_idxs = cutoff_idxs(dataset.get_all_idxs(
                    shuffle(episode_keys[split_point:])), max_len_val)

        print('Train:', len(trn_idxs))
        print('Validation:', len(val_idxs))
        shutil.copytree('.', val_path.joinpath('src'))

        if use_stratification:
            trn_sampler = StratifiedRandomSampler(trn_idxs, idx_class)
        else:
            trn_sampler = SubsetRandomSampler(trn_idxs)
        trn_dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=4, sampler=trn_sampler)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        if use_perceptual_loss:
            perceptual_loss_fn = PerceptualLoss(
                model='net-lin', net='squeeze', use_gpu=use_gpu)
        mse_loss = torch.nn.MSELoss()

        model.train()
        print('Starting training.')

        foci = [0.1, 0.5, 1, 2, 5, 1, 1, 0.7]
        # foci = [0.5, 1]
        # foci = [0.5, 1, 2, 5, 1, 0.7]
        max_epoch = len(foci)
        model.train()
        for epoch in trange(max_epoch):
            # focus = 1
            focus = foci[epoch]
            desc = f'focus={focus}'
            for imgs, masks in tqdm(trn_dataloader, desc=desc):
                imgs = imgs.to(device)
                masks = masks.to(device)

                imgs_hat = model(imgs)

                if mask_mse_loss:
                    loss = masked_mse_loss(imgs_hat, imgs, masks, focus)
                else:
                    loss = mse_loss(imgs_hat, imgs)
                losses = (loss.data.item(),)
                if use_perceptual_loss:
                    scale = 0.1
                    perceptual_loss = scale * perceptual_loss_fn.forward(
                        imgs_hat, imgs).mean()
                    loss += perceptual_loss
                    losses += (perceptual_loss.data.item(),)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tqdm.write(f'Loss: {losses}')

        torch.save(model.state_dict(), save_path.joinpath('model.pth'))
        torch.save(trn_idxs, save_path.joinpath('trn_idxs.pth'))
        torch.save(val_idxs, save_path.joinpath('val_idxs.pth'))
    else:
        model.load_state_dict(torch.load(load_path.joinpath('model.pth')))
        trn_idxs = torch.load(load_path.joinpath('trn_idxs.pth'))
        val_idxs = torch.load(load_path.joinpath('val_idxs.pth'))

    if validate_autoencoder:
        val_num = 4*batch_size if cheat else max_val_num
        val_sampler = StratifiedRandomSampler(
            trn_idxs if cheat else val_idxs, idx_class)
        val_sampler = SequentialSampler(
            val_sampler.as_unshuffled_list(val_num))
        val_dataloader = DataLoader(
            IndexedDataset(dataset), batch_size=batch_size, num_workers=0, sampler=val_sampler)

        model = Inspect(enc, dec).to(device)

        reps = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (idx, (imgs, _)) in enumerate(tqdm(val_dataloader)):
                imgs = imgs.to(device)

                imgs_hat = model(imgs)
                reps.append(model.last_rep.cpu().view(
                    imgs.shape[0], -1).numpy())
                labels.extend(idx_class(idx))

                grid_img = make_grid(
                    torch.cat((imgs.cpu(), imgs_hat.cpu())), nrow=imgs.shape[0])
                tensor2pil(grid_img).save(val_path.joinpath(f'{i}.png'))

        # from tsnecuda import TSNE
        from MulticoreTSNE import MulticoreTSNE as TSNE
        from sklearn.decomposition import PCA
        reps = np.vstack(reps)
        # reps = PCA(n_components=2, copy=False).fit_transform(reps)
        reps = TSNE(n_components=2, perplexity=30,
                    learning_rate=10, n_jobs=10).fit_transform(reps)

        for label in set(labels):
            idxs = [i for i, x in enumerate(labels) if x == label]
            plt.scatter(reps[idxs, 0], reps[idxs, 1], label=label, marker='.')

        plt.rc('font', family='serif')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.legend()
        plt.savefig(val_path.joinpath('tsne.png'), dpi=400)

    # Do prediction
    predictor = EncoderPredictor(300, enc)


