import torch

def Perception(img_dim, out_features):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(img_dim[0], 32, 4, 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 4, 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 4, 2),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(256 * img_dim[1] * img_dim[2], out_features)
    )


class InversePerception(torch.nn.Module):
    def __init__(self, img_dim, out_features):
        nn = torch.nn
        self.to_img = nn.Linear(out_features, 256 * img_dim[1] * img_dim[2])
        self.img_dim = img_dim

        self.conv = nn.Sequential(
            nn.Conv2d(256, 128, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 5, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 6, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, img_dim[0], 6, 2),
        )

    def forward(self, x):
        img = self.to_img(x)
        img = torch.reshape(img, (256, img_dim[1], img_dim[2]))
        return self.conv(img)
