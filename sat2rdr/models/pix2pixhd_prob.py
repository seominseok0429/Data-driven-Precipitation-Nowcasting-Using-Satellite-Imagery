import torch
import torch.nn as nn

from functools import partial

def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=True)
    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)
    return layer

def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d
    elif type == 'replication':
        layer = nn.ReplicationPad2d
    elif type == 'zero':
        layer = nn.ZeroPad2d
    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))
    return layer

class Generator(nn.Module):
    def __init__(self,
            input_ch = 3,
            output_ch = 1,
            n_gf = 64,
            norm_type = "InstanceNorm2d",
            padding_type = "reflection",
            n_downsample = 4,
            n_residual = 9,
            ):
        super(Generator, self).__init__()

        act = nn.ReLU(inplace=True)
        self.act = nn.ReLU(inplace=True) # nn.Tanh() (~1 - 1)
        self.sigmoid = nn.Sigmoid()
        input_ch = input_ch
        n_gf = n_gf
        norm = get_norm_layer(norm_type)
        output_ch = output_ch
        pad = get_pad_layer(padding_type)

        model = []
        model += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0), norm(n_gf), act]

        for _ in range(n_downsample):
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1, stride=2), norm(2 * n_gf), act]
            n_gf *= 2

        for _ in range(n_residual):
            model += [ResidualBlock(n_gf, pad, norm, act)]
        self.model = nn.Sequential(*model)

        n_gf_tmp = n_gf

        model_recon = []

        for _ in range(n_downsample):
            model_recon += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                      norm(n_gf//2), act]
            n_gf //= 2

        model_recon += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model_recon = nn.Sequential(*model_recon)

        model_prob = []
        n_gf = n_gf_tmp

        for _ in range(n_downsample):
            model_prob += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                      norm(n_gf//2), act]
            n_gf //= 2

        model_prob += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model_prob = nn.Sequential(*model_prob)

        self.init_weights()
        print(self)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def init_weights(self):
        self.model.apply(self.weights_init)

    def weights_init(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = self.model(x)
        out1 = self.model_recon(out)
        out2 = self.model_prob(out)

        return self.act(out1), self.sigmoid(out2)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

if __name__ == "__main__":
    inputs = torch.randn(2, 4, 300,250)
    G = Generator(input_ch=4)
    outputs, outputs2 = G(inputs)
    print(outputs.shape, outputs2.shape)
