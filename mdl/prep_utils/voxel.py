import pandas as pd
import numpy as np
import math
import logging
from itertools import product
from scipy.spatial.distance import cdist

pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
np.set_printoptions(precision=3, threshold=np.inf, linewidth=160)

DTYPE = 'float16'
DTYPE_INT = 'int32'


METALS = tuple(
    (3, 4, 11, 12, 13) + tuple(range(19, 32))
    + tuple(range(37, 51)) + tuple(range(55, 84))
    + tuple(range(87, 104))
)


def crop_grid(df, cname, nv, vs):
    ''' Slice coords by pfield box size. '''
    bs = nv * vs  # size of pfield box
    coords = df[['x', 'y', 'z']].to_numpy()
    # df['mask'] = [all(x < bs/2 and x >= -bs/2  for x in row) for row in coords]
    df['mask'] = [all(x < bs/2 - vs/2 and x >= -bs/2  for x in row)
                  for row in coords]
    df_cropped = df[df['mask']]
    df_cropped.drop('mask', inplace=True, axis=1)

    # Print error message when slicing looses ligand nodes.
    if not all(df['mask']):
        logging.info(f'[{cname}] Warning: some nodes cropped with image size {bs}')

    return df_cropped


# ------------------------------------
# --- Generate point field voxels. ---
# ------------------------------------
def gen_pfield(df, mol_type, cname, ch_type, nv, vs):
    '''Generate ligand pfield and receptor pfield according to ch_type.
       pfield_phar(atom)[0]: channels separated
       pfield_phar(atom)[1]: channels merged'''
    df_phar = df.copy()
    # df_phar = df[df['ch_type'] == 'phar']
    # df_atom = df[df['ch_type'] == 'atom']

    if ch_type == 'phar':
        pfield = gen_sub_pfield(df_phar, mol_type, ch_type, nv, vs, cname)
        pfield_vol = gen_sub_pfield(df_phar, mol_type, ch_type, nv, vs, cname, mer=True)
        pfield_mer = np.concatenate((pfield, pfield_vol))
    elif ch_type == 'atom':
        raise NotImplementedError('ch_type atom not implemented yet')

    nc = pfield_mer.shape[0]
    nn = 2 * len(df_phar)  # NOTE(v5): + len(df_atom) --> + len(df_phar)
    nn_out = np.count_nonzero(pfield_mer)
    # print(f'[{cname}] ({nn_out}/{nn} atoms; {nc} channels) generated.')

    return pfield_mer


# NOTE: mar and w_mer of original parameters removed for simple applications.
def gen_sub_pfield(df, mol_type, ch_type, nv, vs, cname, mer=False):
    ''' Generate pfield by obtaining the indices of mol df:
        Channels of fields correspond to mol_types.
        Either of pfields with separated or merged channels generated.
    '''
    cname = f'{cname}_{ch_type}'

    # Generate indices from coords and add to df.
    df = get_grid_points(df, nv, vs, cname)

    node_types = node_types_ref(mol_type, ch_type)

    nc = len(node_types)
    pfield = np.zeros((nc, nv, nv, nv), dtype=DTYPE)

    for ch, node_type in enumerate(node_types):
        inds = df[df['node_type'] == node_type][['ix', 'iy', 'iz']].to_numpy()
        inds = tuple(zip(*inds))  # Make each of x, y, z a tuple.
        if inds:
            pfield[ch][inds] = 1

    # Merge channels (node type).
    # Duplication of indices are ignored in summation by np.clip(0, 1).
    if mer:
        pfield = np.clip(np.sum(pfield, axis=0), 0, 1)[np.newaxis, :]

    return pfield


def get_grid_points(df, nv, vs, cname):
    ''' Generate indices from the coordinates of nodes.
        Replace coordinates in df.
    '''
    coords = df[['x', 'y', 'z']].to_numpy()
    points = np.floor(
        coords/vs           # scale pfield by vs 
        - np.modf(nv/2)[0]  # -0.5 for odd nv
        + np.ceil(nv/2)     # shift to [0: nv]
    )
    points = points.astype(DTYPE_INT).tolist()

    # Check if different atoms have the same index.
    vals, cnt = np.unique(points, axis=0, return_counts=True)
    is_duplicated = cnt > 1
    if True in is_duplicated:
        logging.info(f'[{cname}] Warning: points duplicated --> Reduce voxel size!')
        # print(f'{vals[is_duplicated]}')

    df[['ix', 'iy', 'iz']] = pd.DataFrame(points)
    return df


def node_types_ref(mol_type, ch_type):
    ''' Number of output channels determined here by the size of node_type'''
    if ch_type == 'atom':
        node_types = {6:1, 7:2, 8:3, 15:4, (16, 34):5, METALS:6,
                      (1, 5, 9, 17, 35, 36, 53, 54, 33, 154):7}
    elif ch_type == 'phar':
        if mol_type == 'bs':
            node_types = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
        elif mol_type == 'ps':
            node_types = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
        else:
            raise ValueError(f'mol_type {mol_type} not recognized')
    return node_types


# --------------------------------------
# --- Generate density field voxels. ---
# --------------------------------------
def gen_dfield(pfield, nv, deff, dfield_type, norm_type, ch_type):
    nc = pfield.shape[0]
    wgt_ch = np.ones((nc))

    dfield = []
    for ch in range(nc):
        dfield_ch = gen_dfield_ch(
            wgt_ch[ch] * pfield[ch], nv, deff, dfield_type
        )
        dfield.append(dfield_ch)

    # Normalize density field.
    # If norm_type == ch, each channel is normalized by its own max val.
    # Else, channels of the same type normalized by common max val.
    # E.g., the last (volume) channel normalized separately.
    if norm_type == 'ch':
        for ch in range(nc):
            dfield[ch] = normalize_dfield(dfield[ch])
    elif norm_type == 'field' and ch_type == 'phar':
        dfield[:nc-1] = normalize_dfield(dfield[:nc-1])
        dfield[nc-1] = normalize_dfield(dfield[nc-1])

    return np.array(dfield)


def gen_dfield_ch(pfield_ch, nv, deff, dfield_type):
    '''Return density field for each channel of grid feature'''
    # Get coords of nonzero grids.
    xyzs = np.nonzero(pfield_ch)
    node_vals = pfield_ch[xyzs]
    # Take the array of coords of all nodes to deal with each node.
    xyzs = np.transpose(xyzs)
    # n_atoms = xyzs.shape[0]

    # Initialize density field of whole grid box.
    dfield = np.zeros((nv, nv, nv))

    # NOTE: Generate distance subgrid (of distances to the subgrid center O).
    dist = gen_dist_grid(deff)

    # Sum over fields of each node using dist subgrid and 
    for idx, xyz in enumerate(xyzs):
        if dfield_type == 1:
            rw = 1.5  # Van der Waals radius
            subgrid = node_vals[idx] * np.exp(-0.5*dist**2/rw**2)
        elif dfield_type == 2:
            rw = 1.5
            with np.errstate(divide='ignore'):
                subgrid = node_vals[idx] * (1 - np.exp(-1.0*(rw/dist)**12))
        elif dfield_type == 3:
            rw = 1.5
            with np.errstate(divide='ignore'):
                subgrid = node_vals[idx] * (1 - np.exp(-2.0*(rw/dist)**6))
        else:
            raise ValueError('Unknown dfield type: 1, 2, 3 only')

        subgrid[dist > deff] = 0.0
        # NOTE: place subgrid into whole grid centerd at each node(xyz). 
        dfield_node = place_subgrid(xyz, subgrid, nv, deff)
        dfield += dfield_node

    # dfield = dfield.reshape(1, -1).T
    # nfactor = 1.0/(rw**3*(2*np.pi)**1.5)
    nfactor = 1.0
    dfield = nfactor * dfield.reshape(nv, nv, nv)

    return dfield


def gen_dist_grid(deff):
    '''Generate distance subgrid:
       (1) Value of each voxel is the distance from the voxel to subgrid center.
       (2) Replace parts of whole grid whose centers are the pfield nodes.
    '''
    dmax = math.ceil(deff) + 1  # 1 is buffer of applying deff
    grid_size = 2 * dmax + 1

    grid_x = np.arange(-dmax, dmax + 1)
    grid = np.array(list(product(grid_x, grid_x, grid_x)))

    xyz = np.zeros((1, 3))
    dist = cdist(grid, xyz).reshape(grid_size, grid_size, grid_size)

    return dist


def place_subgrid(xyz, subgrid, nv, deff):
    '''Place subgrid centerd at given node (xyz).'''
    dmax = math.ceil(deff) + 1  # 1 is buffer of applying deff
    nv_eff = nv + 2*dmax
    grid = np.zeros((nv_eff, nv_eff, nv_eff))
    grid[xyz[0]: xyz[0] + 2*dmax + 1,
         xyz[1]: xyz[1] + 2*dmax + 1,
         xyz[2]: xyz[2] + 2*dmax + 1] = subgrid
    grid = grid[dmax: dmax + nv, dmax: dmax + nv, dmax: dmax + nv]

    return grid


def normalize_dfield(dfield):
    max_val = np.max(dfield)
    if max_val:
        dfield = dfield/max_val
    return dfield


def save_dfield(features, dfield_file):
    '''Save rec-lig dfields and smiles dictionay into npz.'''
    np.savez_compressed(dfield_file, **features)  # NOTE: ** used. 
    logging.info(f'[{dfield_file}] Density field generated with normalization.')
