import torch

class Conv2dAuto(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class ConvTranspose2dAuto(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.output_padding = (1, 1)


def BNorm(*layers):
    return torch.nn.Sequential(
        *layers,
        torch.nn.BatchNorm2d(layers[0].out_channels)
    )


def get_total_stride(model):
    stride = 1

    for layer in model.children():
        if isinstance(layer, torch.nn.Sequential):
            stride *= get_total_stride(layer)
        elif isinstance(layer, torch.nn.modules.conv._ConvNd):
            assert all(x == layer.stride[0] for x in layer.stride)
            stride *= layer.stride[0]

    return stride
