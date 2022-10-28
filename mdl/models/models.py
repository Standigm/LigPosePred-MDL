import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from se3cnn.image.gated_block import GatedBlock


# Tensor precision in print
torch.set_printoptions(precision=2, sci_mode=False)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False,
                                       track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            f'normalization layer {norm_type} not found')
    return norm_layer


def get_scheduler(optimizer, arg):
    if arg.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - arg.ne) / float(arg.ne_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif arg.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=arg.lr_decay_iters, gamma=0.1)
    elif arg.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif arg.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=arg.ne, eta_min=0)
    else:
        return NotImplementedError(
            f'learning rate policy {arg.lr_policy} not implemented')
    return scheduler


def update_learning_rate(scheduler, optimizer):
    '''Called once every epoch'''
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    # print(f'learning rate = {lr:.7f}')

    return lr


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    f'Initialization method {init_type} not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('> Initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if all(x > -1 for x in gpu_ids):
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = nn.DataParallel(net, gpu_ids)
    else:
        net.to(torch.device('cpu'))
    init_weights(net, init_type, gain=init_gain)
    return net


def define_d(
    type_d,
    input_nc,
    nkd=4,
    out_dim=1,
    norm='batch',
    init_type='normal',
    init_gain=0.02,
    dropout=0.1,
    gpu_ids=[]
):
    '''Define Discriminator (PatchGAN only).'''
    net = None

    if type_d == 'se3cnn':
        net = SE3CNN(
            nc=input_nc,
            nk=nkd,
            out_dim=out_dim
        )
    elif type_d == 'se3cnn2':
        net = SE3CNN2(
            nc=input_nc,
            nk=nkd,
            out_dim=out_dim
        )
    # elif type_d == 'cnn':
    #     net = CNN(input_nc, nkd, ks)
    else:
        raise NotImplementedError(
            f'Discriminator type {type_d} not recognized')

    return init_net(net, init_type, init_gain, gpu_ids)


class AvgSpacial(nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        # [batch, features]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)


class SE3CNN(nn.Module):
    def __init__(self, nc, nk, out_dim):
        '''Parameters:
           - representation multiplicities (scalar, vector and dim. 5 repr.) for input|output
           - non linearities for scalars and gates (None for no non-linearity)
           - stride, padding, etc.
        '''
        super(SE3CNN, self).__init__()

        ks = 4
        # nkf = nc
        nkf = nk * 4 * 3
        # NOTE: choice of multiplicities is completely arbitrary
        features = [
            (nc, ),  # input (scalar field)
            (nk, nk, nk, 1),
            (nk*2, nk*2, nk*2, 0),
            (nk*2, nk*2, nk*2, 0),
            (nk*2, nk*2, nk*2, 0),
            (nk*4, nk*4, nk*4, 0),
            (nk*4, nk*4, nk*4, 0),
            (nk*4, nk*4, nk*4, 0),
            (nkf, )  # scalar fields to end with fully-connected layers
        ]

        pd = ks//2 - 1 if ks % 2 == 0 else ks//2
        common_block_params = {
            'size': ks,
            'padding': pd
        }

        block_params = [
            {'activation': (None, torch.sigmoid), 'stride': 1},
            {'activation': (F.relu, torch.sigmoid), 'stride': 2},
            {'activation': (F.relu, torch.sigmoid), 'stride': 1},
            {'activation': (F.relu, torch.sigmoid), 'stride': 1},
            {'activation': (F.relu, torch.sigmoid), 'stride': 2},
            {'activation': (F.relu, torch.sigmoid), 'stride': 1},
            {'activation': (F.relu, torch.sigmoid), 'stride': 1},
            {'activation': None, 'stride': 1},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1],
                             **common_block_params, **block_params[i])
                  for i in range(len(block_params))]

        self.seq1 = nn.Sequential(*blocks)
        self.seq2 = nn.Sequential(
            AvgSpacial(),
            nn.Linear(nkf, 50),
            nn.ReLU(),
            nn.Linear(50, out_dim),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, inp):  # pylint: disable=W
        '''param inp: [batch, features, x, y, z] '''
        x1 = self.seq1(inp)
        # print(inp.size())
        # print(x1.size())
        x2 = self.seq2(x1)  # [batch, features]

        return x2


class SE3CNN2(nn.Module):
    def __init__(self, nc, nk, out_dim, norm='batch', inorm=1):
        '''Parameters:
           - representation multiplicities (scalar, vector and dim. 5 repr.)
             for input|output
           - non linearities for scalars and gates (None for no non-linearity)
           - stride, padding, etc.
        '''
        super(SE3CNN2, self).__init__()

        norm = None
        ct = 'normal'  # convolution type (custom|normal)
        ks = 4
        nkf = nc
        self.down1 = SE3Down(ct, (nc,), (nk*1, nk*1, nk*1, 1), ks, sf=1,
                             act=(False, True), norm=norm)
        self.down2 = SE3Down(ct, (nk*1, nk*1, nk*1, 1), (nk*2, nk*2, nk*2, 0), ks, sf=1,
                             act=(True, True), norm=norm)
        self.down3 = SE3Down(ct, (nk*2, nk*2, nk*2, 0), (nk*4, nk*4, nk*4, 0), ks, sf=1,
                             act=(True, True), norm=norm)
        self.down4 = SE3Down(ct, (nk*4, nk*4, nk*4, 0), (nkf,), ks, sf=1,
                             act=(False, False), norm=norm)

        self.seq2 = nn.Sequential(
            AvgSpacial(),
            nn.Linear(nkf, 50),
            nn.ReLU(),
            nn.Linear(50, out_dim),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        out_down1 = self.down1(x)
        out_down2 = self.down2(out_down1)
        out_down3 = self.down3(out_down2)
        out_down4 = self.down4(out_down3)

        out = self.seq2(out_down4)  # [batch, features]

        return out_down4, out


class SE3Down(nn.Module):
    ''' SE3 downsampling block for Unet|Resnet'''

    def __init__(self, conv_type, in_feature, out_feature, ks, sf,
                 out_ng=None, act=(True, True),
                 norm='batch', bias=False, do=None):
        ''' Parameters:
            - conv_type:
                custom (Conv3d + Interpolation) for fractional scaling 
                normal (Conv3d with stride)
            - out_ng: output # grids for custom sizing, ks: kernel_size,
        - sf: scale_factor (used as stride),
        - norm: normalization (batch, instance, None)
        - do: dropout, bias: use bias
        '''
        super(SE3Down, self).__init__()
        # Padding
        pd = ks//2 - 1 if ks % 2 == 0 else ks//2

        # Params for SE3 GatedBlock
        block_params = {'size': ks, 'stride': sf, 'padding': pd,
                        'normalization': norm, 'bias': bias,
                        'capsule_dropout_p': do}
        if act == (True, True):
            block_params['activation'] = (F.relu, torch.sigmoid)
        elif act == (False, True):
            block_params['activation'] = (None, torch.sigmoid)
        elif act == (False, False):
            block_params['activation'] = (None, None)

        layers = []
        if conv_type == 'custom' or sf == 1:
            # Downsample by fractional scale to reduce voxel info loss
            # e.g., (2/3)**3 = 8/27 ~ 1/4 = (1/2)**2 of pixel case
            # Interpolate(scale_factor=2/3, mode='trilinear'),
            if ks % 2 == 0:
                layers += [nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0)]
            block_params['stride'] = 1
            layers += [GatedBlock(in_feature, out_feature, **block_params)]

            # NOTE: size is used instead of scale_factor for up-down matching
            layers += [Interpolate(scale_factor=1/sf, mode='trilinear')]
            # layers += [Interpolate(size=(out_ng, out_ng, out_ng),
            #                        mode='trilinear')]

        elif conv_type == 'normal':
            layers += [GatedBlock(in_feature, out_feature, **block_params)]

        else:
            raise NotImplementedError(
                f'Convolution type {conv_type} not implemented')

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Conv(nn.Module):
    ''' Conv block without down|upsampling for Unet|Resnet'''

    def __init__(self, in_nc, out_nc, ks, nl, bias=False, do=0.0):
        super(Conv, self).__init__()
        layers = []
        if ks % 2 == 0:
            layers += [nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0)]
        pd = ks//2 - 1 if ks % 2 == 0 else ks//2
        layers += [nn.Conv3d(in_nc, out_nc, ks, 1, pd, bias=bias)]
        layers += [nl(out_nc)] if nl else []
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(do)] if do else []

        self.seq = nn.Sequential(*layers)

    def forward(self, x, skip_input=None):
        x = self.seq(x)
        # For UnetGenerator
        if skip_input != None:
            x = torch.cat((x, skip_input), 1)
        return x


class Interpolate(nn.Module):
    ''' Replace stride in down|upsampling for:
        - fractional scaling
        - avoiding checkerboard patterns 
        - consistency between down|up 
    '''

    def __init__(self, mode, size=None, scale_factor=None):
        super(Interpolate, self).__init__()
        self.interpolate = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interpolate(x, size=self.size,
                             scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=True,
                             recompute_scale_factor=True)
        return x
