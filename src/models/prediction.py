import torch


class Predictor(torch.nn.Module):
    def __init__(self, feature_size):
        super(Predictor, self).__init__()
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
        '''
        To do batches, x.shape[0] = sum(len(objects) for objects in batch).
        Flatten it.
        '''
        x = x.view(x.shape[0], -1)
        return self.mlp(x)


class EncoderPredictor(torch.nn.Module):
    def __init__(self, feature_size, enc):
        super(EncoderPredictor, self).__init__()
        self.predictor = Predictor(feature_size)

        # Freeze encoder weights
        for param in enc.parameters():
            param.requires_grad = False
        self.encoder = enc
        self.encoder.eval()

    def forward(self, x):
        encoding = self.encoder(x)
        prediction = self.predictor(encoding)
        return prediction.view(x.shape[0], 1, 15, 20)
