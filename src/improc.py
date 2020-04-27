import torch
import numpy as np
from torchvision import transforms

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

def extract_segment(img, seg, obj_id):
    mask = torch.eq(seg, obj_id)
    return torch.where(mask, img, torch.tensor(0.0)), mask


def get_individual_segments(screens, segmaps):
    imgs = []
    for screen, segmap in zip(screens, segmaps):
        # TODO: Get unique from the npz
        for obj_id in torch.unique(segmap):
            img, _ = extract_segment(screen, segmap, obj_id)
            imgs.append(img)

    return torch.stack(imgs)


def torch_to_normal_shape(img):
    if torch.is_tensor(img):
        return img.permute(1, 2, 0)

    return np.moveaxis(img, 0, 2)


if __name__ == '__main__':
    from PIL import Image
    from matplotlib import pyplot as plt

    img = pil2tensor(Image.open('/home/nistath/Desktop/run1/images/4_300_screen.png'))
    seg = torch.from_numpy(np.array(Image.open('/home/nistath/Desktop/run1/images/4_300_labels.png')))

    out, _ = extract_segment(img, seg, 0)
    out = np.moveaxis(out.numpy(), 0, 2)
    plt.imshow(out)
    plt.show()

