import torch
import torch.nn.functional as F
import numpy as np
import scipy as sp
from scipy.stats import special_ortho_group as so
from math import sin, cos

DTYPE = 'float32'


class ArrayToXyzv():
    ''' Transform 3d array to xyzv (coordinates and value).  '''
    def __init__(self, th=0.0):
        self.th = th

    def __call__(self, x):
        xyzs = np.where(x > self.th)
        values = x[xyzs]
        xyzs = np.array(xyzs)

        return xyzs, values


class XyzvToArray():
    ''' Transform xyzv to 3d array.'''
    def __init__(self, nv):
        self.nv = nv

    def __call__(self, xyzs, values):
        # mask in-box points
        xyzs = np.rint(xyzs).astype(int)
        col_mask = (xyzs < self.nv).all(axis=0)
        xyzs = xyzs[:, col_mask]
        # mask in-box values 
        values = values[col_mask]

        array = np.zeros((self.nv, self.nv, self.nv))
        array[tuple(xyzs)] = values

        return array


class RotateVoxel():
    ''' Apply random|fixed rotation to given 3d array.
        Parameters (for fixed rotation): Euler angles (a, b, c)
    '''
    def __init__(self, euler_angles=None):
        if euler_angles and len(euler_angles) not in (0, 3):
            raise ValueError(
                'Euler angles shoud be a list of three angle values.')
        else:
            self.euler_angles = euler_angles

    def get_rotation(self):
        ''' Return random or fixed rotation matrix.'''
        if self.euler_angles:  # Fixed rotation
            a = self.euler_angles[0]
            b = self.euler_angles[1]
            c = self.euler_angles[2]
            ca, cb, cc = cos(a), cos(b), cos(c)
            sa, sb, sc = sin(a), sin(b), sin(c)
            rot = np.asarray(
                [[ca*cb, ca*sb*sc-sa*cc, ca*sb*cc-sa*sc],
                 [sa*cb, sa*sb*sc-ca*cc, sa*sb*cc-ca*sc],
                 [-sb, cb*sc, cb*cc]])
        else:  # Random rotation
            sp.random.seed()
            rot = so.rvs(3)

        return rot

    def __call__(self, array):
        ''' Rotate 1+3d array by random|fixed roatation matrix rot.'''
        # Get random|fixed rotation.
        rot = self.get_rotation()

        nc = array.shape[0]
        nv = array.shape[1]
        array_rot = np.zeros((nc, nv, nv, nv))
        for ch in range(nc):
            # Get xyz and values from 3d array of each channel.
            xyzs_ch, values_ch = ArrayToXyzv(th=0.0)(array[ch])
            # Apply random|fixed rotation to 3d array.
            xyzs_ch_rot = rot @ (xyzs_ch - nv//2) + nv//2
            # Retrieve 3d array from rotated xyz and values.
            array_rot[ch] = XyzvToArray(nv=nv)(xyzs_ch_rot, values_ch)

        return array_rot.astype(DTYPE)


class Reshape3D():
    ''' Reshape 3d array (Note: for csv).'''
    def __init__(self, n_channel, n_grid):
        self.n_channel = n_channel
        self.n_grid = n_grid

    def __call__(self, array):
        array = array.reshape(
            [self.n_channel, self.n_grid, self.n_grid, self.n_grid])

        return array 


class Resize3D():
    ''' Resize 3d array.'''
    def __init__(self, target_size, interpolate_mode='cubic'):
        self.target_size = target_size
        self.interpolate_mod = interpolate_mode

    def __call__(self, array):
        return F.interpolate(array, self.target_size)


class ToTensor():
    ''' Convert ndarray to torch tensor.'''
    def __init__(self):
        pass

    def __call__(self, array):
        return torch.from_numpy(array).float()


class Normalize():
    ''' To be modified to channel dependent normalization.'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        # x = (x - x.mean())/x.std()
        x = (x - self.mean)/self.std

        return x


class InvNormalize():
    ''' To be modified to channel dependent normalization.'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = x * self.std + self.mean

        return x


class AtomRound():
    ''' Note: Check usage.'''
    def __init__(self, th, lb, ub):
        self.th = th
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        x[x >= self.th] = self.ub
        x[x <= self.th] = self.lb
        # x[(x < self.th) & (x > -self.th)] = 0.0

        return x
