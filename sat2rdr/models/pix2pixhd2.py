import torch
import torch.nn as nn

from functools import partial

from .pos_emb import Time_MLP
from .midblock import (ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


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
            act_type = 'silu',
            mid_type = 'resnet',
            time_emb = None
            ):
        super(Generator, self).__init__()
        if act_type == 'silu':
            act = nn.SiLU()
        elif act_type == 'gelu':
            act = nn.GELU()
        else:
            act = nn.ReLU(inplace=True)

        input_ch = input_ch
        n_gf = n_gf
        norm = get_norm_layer(norm_type)
        output_ch = output_ch
        pad = get_pad_layer(padding_type)

        self.time_emb = time_emb
        if time_emb is not None:
            self.day_emb = Time_MLP(dim=64)
            self.hour_emb = Time_MLP(dim=64)

        encoder = []
        mid = []
        decoder = []

        for _ in range(n_downsample):
            encoder += [nn.Conv2d(input_ch, n_gf, kernel_size=3, padding=1, stride=2), norm(2 * n_gf), act]
            n_gf *= 2
        
        for _ in range(n_residual):
            if mid_type == 'resnet':
                mid += [ResidualBlock(n_gf, pad, norm, act)]
            elif mid_type == 'van':
                mid += [VANSubBlock(n_gf, mlp_ratio=8, drop=0, drop_path=0, act_layer=nn.GELU)]
            elif mid_type == 'convnext':
                mid += [ConvNeXtSubBlock(n_gf,  mlp_ratio=8, drop=0, drop_path=0)]
            else:
                print('model error')

        for _ in range(n_downsample):
            decoder += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                      norm(n_gf//2), act]
            n_gf //= 2

        decoder += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]

        self.encoder = nn.Sequential(*encoder)
        self.mid = nn.Sequential(*mid)
        self.decoder = nn.Sequential(*decoder)

        self.init_weights()
        print(self)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def init_weights(self):
        self.encoder.apply(self.weights_init)
        self.mid.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

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

    def forward(self, x, day=None, hour=None):

        if self.time_emb is not None:
            day = self.day_emb(day)
            hour = self.hour_emb(hour)

        x = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x)
        return x


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
    G = Generator(input_ch=4, mid_type='van')
    outputs = G(inputs)
    print(outputs.shape)

