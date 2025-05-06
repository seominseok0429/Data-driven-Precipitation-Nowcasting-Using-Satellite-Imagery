import torch
from torch import nn

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        diff_log = torch.log(target+1) - torch.log(pred+1)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class GANLoss(object):
    def __init__(self, args,
            n_D = 1):
        self.gpu_id = args.gpu_id
        self.cc_loss = args.cc_loss
        #self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.device = 'cuda'
        self.dtype = torch.float32

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.lambda_FM = 10
        self.n_D = n_D

    def _discriminate(self, input_label, test_image, D):

        input_concat = torch.cat((input_label, test_image.detach()), dim=1)

        return D(input_concat)

    def _ganLoss(self, inputs, is_real=True):
        grid = get_grid(inputs[0][-1], is_real=is_real).to(self.device, self.dtype)
        loss = self.criterion(inputs[0][-1], grid)
        return loss

    def _Inspector(self, target, fake):

        rd = target - torch.mean(target)
        fd = fake - torch.mean(fake)

        r_num = torch.sum(rd * fd)
        r_den = torch.sqrt(torch.sum(rd ** 2)) * torch.sqrt(torch.sum(fd ** 2))
        PCC_val = r_num/(r_den + 1e-6)

        numerator = 2*PCC_val*torch.std(target)*torch.std(fake)
        denominator = (torch.var(target) + torch.var(fake)
                        + (torch.mean(target) - torch.mean(fake))**2)

        CCC_val = numerator/(denominator + 1e-6)
        loss_CC = (1.0 - CCC_val)

        return loss_CC


    def _matchingLoss(self, pred_real, pred_fake):
        loss_G_GAN_Feat = 0
        for i in range(len(pred_fake)-1):
            loss_G_GAN_Feat += self.FMcriterion(pred_fake[i], pred_real[i].detach())

        return loss_G_GAN_Feat


    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)

        #fake detection and loss
        pred_fake_pool = self._discriminate(input, fake, D)
        loss_D_fake = self._ganLoss(pred_fake_pool, False)

        #real detection and loss
        pred_real = self._discriminate(input, target, D)
        loss_D_real = self._ganLoss(pred_real, True)

        #GAN loss(fake passability loss)
        pred_fake = D(torch.cat((input, fake), dim=1))
        loss_G = self._ganLoss(pred_fake, True)

        loss_fm = self._matchingLoss(pred_real, pred_fake)

        loss_D = (loss_D_fake + loss_D_real)*0.5

        if self.cc_loss:
            loss_CC = self._Inspector(target, fake)
            loss_G = (loss_fm*10.0 + loss_G) + loss_CC*5
            print('h')
        else:
            loss_G = (loss_fm*10.0 + loss_G)

        return loss_D, loss_G, target, fake
