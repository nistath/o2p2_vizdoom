import torch
from models.conv import Conv2dAuto, ConvTranspose2dAuto, BNorm, get_total_stride


__all__ = ['Perception', 'InversePerception', 'ConvAutoencoder']


class View(torch.nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def Perception(img_dim, out_features):
    nn = torch.nn

    conv = nn.Sequential(
        Conv2dAuto(img_dim[0], 32, 3, 2),
        nn.ReLU(inplace=True),
        BNorm(Conv2dAuto(32, 64, 3, 2)),
        nn.ReLU(inplace=True),
        BNorm(Conv2dAuto(64, 128, 3, 2)),
        nn.ReLU(inplace=True),
        Conv2dAuto(128, 256, 3, 2),
        nn.ReLU(inplace=True),
    )

    # Flatten into out_features
    stride = get_total_stride(conv)
    final_size = img_dim[1] * img_dim[2] // stride**2

    return nn.Sequential(
        conv,
        nn.Flatten(),
        nn.Linear(256 * final_size, out_features)
    )


def ConvAutoencoder(img_dim, have_linear=False):
    nn = torch.nn

    encoder_suffix = [Conv2dAuto(128, 1, 3, 2),]
    if have_linear:
        encoder_suffix[0] = BNorm(encoder_suffix[0])
        encoder_suffix += [
            View(-1, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300),
            View(-1, 1, 15, 20),
        ]

    encoder_suffix.append(nn.Tanh())

    encoder = nn.Sequential(
        Conv2dAuto(img_dim[0], 32, 3, 2),
        nn.ReLU(inplace=True),
        BNorm(Conv2dAuto(32, 64, 3, 2)),
        nn.ReLU(inplace=True),
        BNorm(Conv2dAuto(64, 128, 3, 2)),
        nn.ReLU(inplace=True),
        *encoder_suffix
    )

    decoder = nn.Sequential(
        ConvTranspose2dAuto(1, 128, 5, 2),
        nn.ReLU(inplace=True),
        BNorm(ConvTranspose2dAuto(128, 64, 5, 2)),
        nn.ReLU(inplace=True),
        BNorm(ConvTranspose2dAuto(64, 32, 7, 2)),
        nn.ReLU(inplace=True),
        ConvTranspose2dAuto(32, img_dim[0], 5, 2),
        nn.Sigmoid()
    )

    return encoder, decoder


class InversePerception(torch.nn.Module):
    def __init__(self, img_dim, in_features):
        super(InversePerception, self).__init__()
        nn = torch.nn

        self.conv = nn.Sequential(
            BNorm(ConvTranspose2dAuto(256, 128, 5, 2)),
            nn.ReLU(inplace=True),
            BNorm(ConvTranspose2dAuto(128, 64, 5, 2)),
            nn.ReLU(inplace=True),
            BNorm(ConvTranspose2dAuto(64, 32, 5, 2)),
            nn.ReLU(inplace=True),
            ConvTranspose2dAuto(32, img_dim[0], 5, 2),
            nn.Sigmoid()
        )

        stride = get_total_stride(self.conv)
        final_size = img_dim[1] * img_dim[2] // stride**2

        self.to_img = nn.Sequential(
            nn.Linear(in_features, 256 * final_size),
            nn.ReLU(inplace=True)
        )
        self.start_dim = (img_dim[1] // stride, img_dim[2] // stride)

    def forward(self, x):
        img = self.to_img(x)
        img = img.view((img.shape[0], 256, *self.start_dim))
        return self.conv(img)
