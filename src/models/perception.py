import torch
from models.conv import Conv2dAuto, ConvTranspose2dAuto, BNorm, get_total_stride


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


class InversePerception(torch.nn.Module):
    def __init__(self, img_dim, in_features):
        super(InversePerception, self).__init__()
        nn = torch.nn

        agg = img_dim[1] * img_dim[2]
        self.to_img = nn.Sequential(
            nn.Linear(in_features, 16 * in_features),
            nn.ReLU(inplace=True),
            nn.Linear(16 * in_features, 4 * agg // (16 * in_features)),
            nn.ReLU(inplace=True),
            nn.Linear(4 * agg // (16 * in_features), 3 * agg),
            nn.Sigmoid()
        )
        self.img_dim = img_dim

    def forward(self, x):
        img = self.to_img(x)
        return img.view((img.shape[0], 3, self.img_dim[1], self.img_dim[2]))


class InversePerceptionConv(torch.nn.Module):
    def __init__(self, img_dim, in_features):
        super(InversePerceptionConv, self).__init__()
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
