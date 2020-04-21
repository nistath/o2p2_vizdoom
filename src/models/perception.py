import torch


def Perception(img_dim, out_features):
    nn = torch.nn

    final_size = img_dim[1] * img_dim[2] // (2**8)

    return nn.Sequential(
        nn.Conv2d(img_dim[0], 32, 5, 2, 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 5, 2, 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 5, 2, 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 5, 2, 2),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(256 * final_size, out_features),
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

        final_size = img_dim[1] * img_dim[2] // (2**8)

        self.to_img = nn.Sequential(
            nn.Linear(in_features, 256 * final_size),
            nn.ReLU(inplace=True)
        )
        self.start_dim = (img_dim[1] // 16, img_dim[2] // 16)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, 2, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 5, 2, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, img_dim[0], 5, 2, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        img = self.to_img(x)
        img = img.view((img.shape[0], 256, *self.start_dim))
        return self.conv(img)
