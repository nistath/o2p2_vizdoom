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
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f'Using device {device}.')

    experiment_name = datetime.now().isoformat()
    experiment_name += '_predict_pls_work'
    results_path = Path('/home/nistath/Desktop/val/')
    val_path = results_path.joinpath(experiment_name)
    load_path = results_path.joinpath(
        '2020-05-12T13:22:17.994781_pls_work/save')
    save_path = val_path.joinpath('save')

    predictor_path = results_path.joinpath(
        '2020-05-12T13:55:26.403096_predict_pls_work/save/predictor.pth')

    img_shape = (240, 320)
    dataset = DoomSegmentedDataset('/home/nistath/Desktop/run2/states.npz',
                                   '/home/nistath/Desktop/run2/images/', desired_size=img_shape,
                                   #    blacklist=(0, 1,)
                                   )

    mask_mse_loss = True
    use_stratification = True
    use_perceptual_loss = True
    reuse_split = True
    reuse_autoencoder = False  # implies split will be reused
    validate_autoencoder = True

    use_convautoencoder = True
    use_tsne = True

    predict = False
    reuse_predictor = True
    validate_predictor = False

    if reuse_autoencoder:
        if val_path.exists():
            raise ValueError('will not overwrite')
    save_path.mkdir(exist_ok=True, parents=True)
    val_path.mkdir(exist_ok=True, parents=True)

    # whether to predict on the train set
    cheat = False

    split = 0.5
    max_len_trn = 3000
    max_len_val = None
    max_val_num = 1000
    p_max_val_num = 200
    batch_size = 32
    p_batch_size = 32 // 4

    focus_annealing_schedule = [0.1, 0.5, 1, 2, 5, 1, 1, 0.7]

    if use_convautoencoder:
        enc, dec = ConvAutoencoder((3,) + img_shape, have_linear=False)
    else:
        num_features = 256
        enc = Perception((3,) + img_shape, num_features)
        dec = InversePerception((3,) + img_shape, num_features)

    model = torch.nn.Sequential(enc, dec).to(device)

    if not reuse_autoencoder:
        if reuse_split:
            trn_idxs = torch.load(load_path.joinpath('trn_idxs.pth'))
            val_idxs = torch.load(load_path.joinpath('val_idxs.pth'))
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

        max_epoch = len(focus_annealing_schedule)
        model.train()
        for epoch in trange(max_epoch, desc='autoencoder'):
            focus = focus_annealing_schedule[epoch]
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
            IndexedDataset(dataset), batch_size=batch_size, num_workers=4, sampler=val_sampler)

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

        reps = np.vstack(reps)
        if use_tsne:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            reps = TSNE(n_components=2, perplexity=30,
                        learning_rate=10, n_jobs=10).fit_transform(reps)
        else:
            from sklearn.decomposition import PCA
            reps = PCA(n_components=2, copy=False).fit_transform(reps)

        for label in set(labels):
            idxs = [i for i, x in enumerate(labels) if x == label]
            plt.scatter(reps[idxs, 0], reps[idxs, 1], label=label, marker='.')

        plt.rc('font', family='serif')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(val_path.joinpath('tsne.png'), dpi=400,
                    bbox_inches='tight', pad_inches=0)

    # Do prediction
    if not predict:
        exit()

    print('PREDICTION TIME BABY')
    predictor = Predictor(300).to(device)
    encoder = enc
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    decoder = dec
    for param in decoder.parameters():
        param.requires_grad = False
    decoder.eval()

    p_dataset = PredictionDataset(dataset, trn_idxs)
    p_trn_idxs = p_dataset.get_all_idxs()
    if use_stratification:
        p_trn_sampler = StratifiedRandomSampler(p_trn_idxs, idx_class)
    else:
        p_trn_sampler = SubsetRandomSampler(p_trn_idxs)

    p_trn_dataloader = DataLoader(
        p_dataset, batch_size=p_batch_size, num_workers=4, sampler=p_trn_sampler)

    if not reuse_predictor:
        p_mse_loss = torch.nn.MSELoss()
        p_opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

        p_max_epoch = len(focus_annealing_schedule)
        for epoch in trange(p_max_epoch, desc='pred'):
            focus = focus_annealing_schedule[epoch]
            desc = f'focus={focus}'
            for (s_imgs, s_masks), (t_imgs, t_masks) in tqdm(p_trn_dataloader, desc=desc):
                t_masks = t_masks.to(device)
                s_imgs = s_imgs.to(device)
                t_imgs = t_imgs.to(device)

                s_objs = encoder.forward(s_imgs)
                t_objs = encoder.forward(t_imgs)
                t_hat_objs = predictor(s_objs)

                # s_imgs_hat = decoder.forward(s_objs)
                # t_imgs_hat = decoder.forward(t_objs)
                t_hat_imgs_hat = decoder.forward(t_hat_objs)

                obj_loss = p_mse_loss(t_hat_objs, t_objs)
                mask_loss = masked_mse_loss(
                    t_hat_imgs_hat, t_imgs, t_masks, focus=focus)

                losses = (obj_loss.item(), mask_loss.item())

                p_loss = obj_loss + mask_loss
                p_opt.zero_grad()
                p_loss.backward()
                p_opt.step()

                tqdm.write(f'Loss: {losses}')

        torch.save(predictor.state_dict(), save_path.joinpath('predictor.pth'))
    else:
        predictor.load_state_dict(torch.load(predictor_path))

    if validate_predictor:
        p_val_dataset = p_dataset if cheat else PredictionDataset(
            dataset, val_idxs)
        p_val_idxs = p_val_dataset.get_all_idxs()

        p_val_num = 4*p_batch_size if cheat else p_max_val_num
        p_val_sampler = StratifiedRandomSampler(
            p_trn_idxs if cheat else p_val_idxs, idx_class)
        p_val_sampler = SequentialSampler(
            p_val_sampler.as_unshuffled_list(p_val_num))
        p_val_dataloader = DataLoader(
            p_val_dataset, batch_size=p_batch_size, num_workers=4, sampler=p_val_sampler)

        predictor.eval()
        with torch.no_grad():
            for i, ((s_imgs, s_masks), (t_imgs, t_masks)) in enumerate(tqdm(p_val_dataloader)):
                s_imgs = s_imgs.to(device)
                t_imgs = t_imgs.to(device)

                s_objs = encoder.forward(s_imgs)
                t_objs = encoder.forward(t_imgs)
                t_hat_objs = predictor(s_objs)

                s_imgs_hat = decoder.forward(s_objs)
                t_imgs_hat = decoder.forward(t_objs)
                t_hat_imgs_hat = decoder.forward(t_hat_objs)

                grid_img = make_grid(
                    torch.cat((s_imgs.cpu(), s_imgs_hat.cpu(), t_imgs.cpu(), t_hat_imgs_hat.cpu(), t_imgs_hat.cpu())), nrow=s_imgs.shape[0])
                tensor2pil(grid_img).save(val_path.joinpath(f'pred_{i}.png'))
