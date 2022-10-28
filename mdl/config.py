# ## MDL model (SE3CNN) configuration
import mdl
import torch
from pathlib import Path
from typing import List, Tuple, Iterable, Tuple, Union


class MDLConfig():
    r"""
    Args:
    Example::
        >>> config = MDLConfig()
    """

    def __init__(
        self,
        # directory for data and outputs
        data_dir: str,  # required
        output_prefix: str,  # required

        # Common (for prep and voxel prep)
        rec_suffix: str = 'pdb',
        lig_suffix: str = 'sdf',
        overwrite: bool = True,
        # Prep
        doc_type: str = 'fixed',  # fixed|flexible
        # Voxel prep
        edge_type_bs: int = 1,  # edge type for binding site
        edge_type_ps: int = 2,  # edge type for ligand
        edge_type_bp: int = 3,  # edge type for binding site - ligand
        src_idx: int = 1,  # starting index of the nodes of given data
        crop_type: str = 'node',  # node|box
        # node: by min dist to all nodes
        # box: by min dist to boundary nodes
        r_int: float = 3.5,  # radius of interaction between rec and lig
        num_nn: int = 1,  # number of nearest neighbors
        num_nn_bp: int = 3,  # number of nearest neighbors
        r_nn: float = 0.0,  # radius centered at each node to neighbors
        norm_type: str = 'sep',  # sep|mer
        nv: int = 24,  # number of voxels along 1d axis
        vs: float = 1.0,  # voxel size (angstrom)
        deff: float = 3.0,  # effective density distance
        dfield_type: int = 3,  # type of dfield')
        dfield_norm_type: str = 'ch',  # ch|field|whole
        save_voxel: str = 'mer',  # mer|sep
                           # mer: save fields of the same bs to single file
                           # sep: save fields of the same bs to sperate files
        gen_edges_bs: str = 'partial',  # partial|full
        # partial: for nodes without edges only
                                 # full: for all nodes ignoring given edes
        no_edges_bp: bool = False,  # Do not generate edges between bs and ps

        # Feature params
        node_feature_size=18,
        checkpoints: Iterable[str] = [
            "checkpoints/ensemble3/pos_cls_20-40/cp-49-auc89.tar",
            "checkpoints/ensemble3/pos_cls_30-40/cp-35-auc86.tar",
            "checkpoints/ensemble3/pos_cls_30-50/cp-17-auc90.tar",
        ],
        label_th: Tuple[float, float] = (20, 40),
        # Model params (SE3CNN)
        type_d: str = 'se3cnn',
        nkd: int = 8,  # discriminator kernels in the first conv
        out_dim: int = 1,
        num_layers: int = 8,  # NOTE: not applied yet
        dropout: float = 0.1,
        use_focal_loss: bool = False,
        norm: str = 'batch',  # batch|instance
        inorm: int = 1,  # input normalization:
        # 0: [0 1] -> [-1 1], 1: [0 1] -> [0 1], 2: [-1 1] -> [-1 1]

        # Meta-parameters
        batch_size: int = 256,
        num_workers: int = 4,  # The number of data loader workers
        num_processes: int = 1,  # The number of processes in parallel
        device: Union[None, torch.DeviceObjType] = None,

        # Task and Data
        task: str = 'cls',  # cls|reg
        data_type: str = 'aff',  # pos|aff|mol
        node_feature_type: str = 'phar',   # atom|phar
        max_num_ps: int = -1,  # max number of poses per binding site
        balance_ratio: float = -1,  # pos to neg data ratio. -1 for no balance
        label_corr: str = 'pos',  # pos|neg
        max_label: float = -1,  # max label (rmsd)
        rot: bool = False,  # Apply on-line random rotation

        # Miscellanea
        seed: Union[int, None] = None,
        **kwargs
    ):
        self.data_dir = data_dir
        self.output_prefix = output_prefix
        
        self.rec_suffix = rec_suffix
        self.lig_suffix = lig_suffix
        self.overwrite = overwrite
        self.doc_type = doc_type
        self.edge_type_bs = edge_type_bs
        self.edge_type_ps = edge_type_ps
        self.edge_type_bp = edge_type_bp
        self.src_idx = src_idx
        self.crop_type = crop_type
        self.r_int = r_int
        self.num_nn = num_nn
        self.num_nn_bp = num_nn_bp
        self.r_nn = r_nn
        self.norm_type = norm_type
        self.nv = nv
        self.vs = vs
        self.deff = deff
        self.dfield_type = dfield_type
        self.dfield_norm_type = dfield_norm_type
        self.save_voxel = save_voxel
        self.gen_edges_bs = gen_edges_bs
        self.no_edges_bp = no_edges_bp
        self.node_feature_size = node_feature_size
        self.type_d = type_d
        self.nkd = nkd
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_focal_loss = use_focal_loss
        self.norm = norm
        self.inorm = inorm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_processes = num_processes
        self.device = device if device is not None else torch.device("cpu")
        self.task = task
        self.data_type = data_type
        self.node_feature_type = node_feature_type
        self.max_num_ps = max_num_ps
        self.balance_ratio = balance_ratio
        self.label_corr = label_corr
        self.max_label = max_label
        self.rot = rot
        self.seed = seed

        self.checkpoints = checkpoints
        self.label_th = label_th


if __name__ == '__main__':
    conf = MDLConfig()
    print(conf.type_d)
