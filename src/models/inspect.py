import torch


class Inspect(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Inspect, self).__init__()
        self.enc = enc
        self.dec = dec
        self.last_rep = None

    def forward(self, x):
        rep = self.enc(x)
        self.last_rep = rep
        return self.dec(rep)
