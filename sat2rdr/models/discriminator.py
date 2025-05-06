import torch
import torch.nn as nn

from functools import partial

class PatchDiscriminator(nn.Module):
    def __init__(self,
            input_ch=3,
            output_ch=1,
            n_df =64,
            ):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        self.act = nn.Tanh()
        input_channel = input_ch + output_ch
        n_df = n_df
        norm = nn.InstanceNorm2d

        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  #


class Discriminator(nn.Module):
    def __init__(self,
            input_ch=3,
            output_ch=1,
            n_df =64,
            n_D=1):
        super(Discriminator, self).__init__()

        for i in range(n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(input_ch, output_ch, n_df))
        self.n_D = n_D

        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
        return result


if __name__ == "__main__":
    inputs = torch.randn(2,5,224,224)
    D = Discriminator(input_ch=4, output_ch=1)
    outputs = D(inputs)
    print(outputs[0][0].shape, outputs[0][1].shape, outputs[0][2].shape, outputs[0][3].shape, outputs[0][4].shape)
