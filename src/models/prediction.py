import torch


class Prediction(torch.nn.Module):
    def __init__(self, feature_size):
        super(Prediction, self).__init__()
        nn = torch.nn

        self.mlp = nn.Sequential(
            nn.Linear(feature_size, 2 * feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * feature_size, 2 * feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(2 * feature_size, feature_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return self.mlp(x)
