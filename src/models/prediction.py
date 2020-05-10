import torch

class Prediction(torch.nn.Module):
    def __init__(self, feature_size):
        super(Prediction, self).__init__()
        nn = torch.nn

        self.mlp = nn.Sequential(
            # nn.Linear()
        )
