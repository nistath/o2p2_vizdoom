import torch
import numpy as np
from torchvision import transforms, datasets

def extract_segment(img, seg, obj_id):
    return torch.where(torch.eq(seg, obj_id), img, torch.tensor(0.0))


if __name__ == '__main__':
    from PIL import Image
    from matplotlib import pyplot as plt

    pil2tensor = transforms.ToTensor()

    img = pil2tensor(Image.open('/home/nistath/Desktop/run1/images/4_300_screen.png'))
    seg = torch.from_numpy(np.array(Image.open('/home/nistath/Desktop/run1/images/4_300_labels.png')))

    out = extract_segment(img, seg, 0)
    out = np.moveaxis(out.numpy(), 0, 2)
    plt.imshow(out)
    plt.show()

