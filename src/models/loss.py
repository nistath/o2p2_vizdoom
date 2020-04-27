import torch
from collections import namedtuple


def masked_mse_loss(inputs, targets, masks, focus=1):
    total_area = masks.numel() // masks.shape[0]
    obj_area = masks.flatten(start_dim=1).sum(1)
    bg_area = total_area - obj_area

    obj_area.unsqueeze_(-1).unsqueeze_(-1)
    bg_area.unsqueeze_(-1).unsqueeze_(-1)

    masks = bg_area * masks + focus * obj_area * masks.logical_not()
    masks.unsqueeze_(1)

    error = inputs - targets
    error = torch.mul(error, error)
    error = error * masks
    error = error.flatten(start_dim=1).sum(1)  # error per object
    # votes = torch.flatten(masks, start_dim=1).sum(1)  # votes per object
    return torch.mean(error)  # all object are equal


# https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
LossOutput = namedtuple(
    "LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3


class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
