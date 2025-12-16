"""
CGCNN Data Loader
Based on official implementation with enhanced usability

Features:
- Automatic CIF file discovery
- Flexible property file formats (CSV, Excel)
- Automatic train/val/test splitting
- Data validation and preprocessing
"""

import os
import csv
import json
import functools
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure


class GaussianDistance:
    """
    Expands distances using Gaussian basis functions.

    This converts scalar distances into fixed-length feature vectors.
    """

    def __init__(self, dmin=0, dmax=8, step=0.2, var=None):
        """
        Initialize Gaussian distance expansion.

        Parameters
        ----------
        dmin : float
            Minimum interatomic distance
        dmax : float
            Maximum interatomic distance
        step : float
            Step size for Gaussian filter
        var : float, optional
            Variance of Gaussian basis (default: step)
        """
        assert dmin < dmax, f"dmin ({dmin}) must be less than dmax ({dmax})"
        assert dmax - dmin > step, f"dmax - dmin ({dmax - dmin}) must be greater than step ({step})"

        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        """
        Apply Gaussian distance filter.

        Parameters
        ----------
        distances : np.ndarray
            Distance matrix of any shape

        Returns
        -------
        expanded_distance : np.ndarray
            Expanded distance matrix with additional dimension
        """
        return np.exp(
            -(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2
        )


class AtomInitializer:
    """
    Base class for initializing atom feature vectors.
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        """Get feature vector for atom type."""
        if atom_type not in self.atom_types:
            supported = sorted(list(self.atom_types))[:20]  # Show first 20
            raise ValueError(
                f"Element with atomic number {atom_type} is NOT supported by your atom_init.json!\n\n"
                f"Supported elements (first 20): {supported}...\n"
                f"Total supported elements: {len(self.atom_types)}\n\n"
                f"This means your CIF files contain an element that is not in atom_init.json.\n"
                f"Please check your CIF files or update atom_init.json to include element {atom_type}."
            )
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        """Load embedding from state dict."""
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        """Return embedding state dict."""
        return self._embedding


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom features from JSON file.

    The JSON file should map element numbers to feature vectors.
    """

    def __init__(self, elem_embedding_file):
        """
        Parameters
        ----------
        elem_embedding_file : str
            Path to JSON file with element embeddings
        """
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)

        elem_embedding = {
            int(key): value for key, value in elem_embedding.items()
        }
        atom_types = set(elem_embedding.keys())

        super(AtomCustomJSONInitializer, self).__init__(atom_types)

        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    Dataset for crystal structures stored as CIF files.

    Supports flexible data organization and automatic preprocessing.
    """

    def __init__(
        self,
        root_dir,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        random_seed=123
    ):
        """
        Initialize CIF dataset.

        Parameters
        ----------
        root_dir : str
            Directory containing CIF files and property file
        max_num_nbr : int
            Maximum number of neighbors to consider
        radius : float
            Cutoff radius for neighbor search (Angstroms)
        dmin : float
            Minimum distance for Gaussian expansion
        step : float
            Step size for Gaussian expansion
        random_seed : int
            Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius

        assert os.path.exists(root_dir), f'root_dir {root_dir} does not exist!'

        # Load property data
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'

        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Shuffle data
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        # Load atom initialization
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)

        # Initialize Gaussian distance filter
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    # Removed lru_cache to prevent memory issues with large datasets
    def __getitem__(self, idx):
        """
        Get a single data point.

        Returns
        -------
        (atom_fea, nbr_fea, nbr_fea_idx) : tuple
            Graph representation
        target : torch.Tensor
            Target property value
        cif_id : str
            Crystal identifier
        """
        cif_id, target = self.id_prop_data[idx]

        # Load crystal structure
        crystal = Structure.from_file(
            os.path.join(self.root_dir, cif_id + '.cif')
        )

        # Get atom features
        atom_fea = np.vstack([
            self.ari.get_atom_fea(crystal[i].specie.number)
            for i in range(len(crystal))
        ])
        atom_fea = torch.Tensor(atom_fea)

        # Get all neighbors
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        # Build neighbor feature arrays
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f'{cif_id} does not have enough neighbors. '
                    f'Consider increasing radius.'
                )
                # Pad with zeros
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) +
                    [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr[:self.max_num_nbr]))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                )

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)

        # Expand distances using Gaussian basis
        nbr_fea = self.gdf.expand(nbr_fea)

        # Convert to tensors
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id

    @classmethod
    def from_directory(
        cls,
        structure_dir,
        property_file,
        output_dir=None,
        atom_init_file=None,
        **kwargs
    ):
        """
        Create dataset from directory of CIF files and property file.

        This is a convenience method that automatically sets up the
        required directory structure.

        Parameters
        ----------
        structure_dir : str
            Directory containing CIF files
        property_file : str
            CSV or Excel file with structure IDs and properties
        output_dir : str, optional
            Directory to prepare data (default: temp directory)
        atom_init_file : str, optional
            JSON file for atom initialization (default: use built-in)
        **kwargs
            Additional arguments passed to CIFData constructor

        Returns
        -------
        dataset : CIFData
            Initialized dataset
        """
        # This will be implemented to provide easy data loading
        raise NotImplementedError("Use prepare_data_directory first")


def collate_pool(dataset_list):
    """
    Collate function for batching crystal data.

    Parameters
    ----------
    dataset_list : list
        List of data points from CIFData

    Returns
    -------
    batch_data : tuple
        Batched graph data
    batch_target : torch.Tensor
        Batched targets
    batch_cif_ids : list
        List of crystal IDs
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0

    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)

        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)

        batch_target.append(target)
        batch_cif_ids.append(cif_id)

        base_idx += n_i

    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx
        ),
        torch.stack(batch_target, dim=0),
        batch_cif_ids
    )


def get_train_val_test_loader(
    dataset,
    collate_fn=collate_pool,
    batch_size=64,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    return_test=False,
    num_workers=0,
    pin_memory=False,
    train_size=None,
    val_size=None,
    test_size=None,
    random_seed=None,
    shuffle=True,
    **kwargs
):
    """
    Create train/val/test data loaders.

    Compatible with official CGCNN implementation.
    Supports both ratio-based and absolute size-based splitting.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Full dataset to split
    collate_fn : callable
        Collate function for batching (default: collate_pool)
    batch_size : int
        Batch size (default: 64)
    train_ratio : float, optional
        Fraction for training (if train_size not specified)
    val_ratio : float
        Fraction for validation (default: 0.1)
    test_ratio : float
        Fraction for testing (default: 0.1)
    return_test : bool
        Whether to return test loader (default: False)
    num_workers : int
        Number of data loading workers (default: 0)
    pin_memory : bool
        Whether to pin memory (default: False)
    train_size : int, optional
        Absolute number of training samples (overrides train_ratio)
    val_size : int, optional
        Absolute number of validation samples (overrides val_ratio)
    test_size : int, optional
        Absolute number of test samples (overrides test_ratio)
    random_seed : int, optional
        Random seed for reproducible splits (default: None, uses random shuffle)
    shuffle : bool
        Whether to shuffle data before splitting (default: True)
        Set to False for sequential split (not recommended)
    **kwargs : dict
        Additional arguments for compatibility with official implementation
        Supports: 'train_size', 'val_size', 'test_size' passed as kwargs

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader (if return_test=True)
    """
    import random

    # Support official CGCNN kwargs format
    if 'train_size' in kwargs and kwargs['train_size'] is not None:
        train_size = kwargs['train_size']
    if 'val_size' in kwargs and kwargs['val_size'] is not None:
        val_size = kwargs['val_size']
    if 'test_size' in kwargs and kwargs['test_size'] is not None:
        test_size = kwargs['test_size']
    if 'random_seed' in kwargs and kwargs['random_seed'] is not None:
        random_seed = kwargs['random_seed']

    total_size = len(dataset)

    # Calculate sizes
    if train_size is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1, "val_ratio + test_ratio must be < 1"
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1, \
                "train_ratio + val_ratio + test_ratio must be <= 1"
        train_size = int(train_ratio * total_size)

    if test_size is None:
        test_size = int(test_ratio * total_size)

    if val_size is None:
        val_size = int(val_ratio * total_size)

    # Ensure minimum sizes for small datasets
    # Each split should have at least 1 sample if ratio > 0
    if val_ratio > 0 and val_size == 0 and total_size >= 3:
        val_size = 1
        print(f'[Warning] val_size was 0, adjusted to 1 for small dataset')
    if test_ratio > 0 and test_size == 0 and total_size >= 3:
        test_size = 1
        print(f'[Warning] test_size was 0, adjusted to 1 for small dataset')

    # Recalculate train_size to ensure we don't exceed total
    if train_size + val_size + test_size > total_size:
        train_size = total_size - val_size - test_size
        if train_size < 1:
            train_size = 1
            # Reduce val/test if needed
            remaining = total_size - train_size
            val_size = remaining // 2
            test_size = remaining - val_size
        print(f'[Warning] Adjusted split sizes for small dataset: '
              f'train={train_size}, val={val_size}, test={test_size}')

    print(f'[Data Split] Total: {total_size}, Train: {train_size}, '
          f'Val: {val_size}, Test: {test_size}')

    # Create indices
    indices = list(range(total_size))

    # Shuffle indices for random split
    if shuffle:
        if random_seed is not None:
            # Use seed for reproducibility
            random.seed(random_seed)
            print(f'[Data Split] Using random seed: {random_seed}')
        random.shuffle(indices)
        print('[Data Split] Data shuffled randomly before splitting')
    else:
        print('[Data Split] Sequential split (no shuffle) - not recommended')

    # Split indices: train | val | test
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)

    # Handle edge case when val_size is 0
    if val_size > 0:
        val_sampler = SubsetRandomSampler(val_indices)
    else:
        val_sampler = SubsetRandomSampler([])
        print('[Warning] Validation set is empty!')

    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    if return_test:
        if test_size > 0:
            test_sampler = SubsetRandomSampler(test_indices)
        else:
            test_sampler = SubsetRandomSampler([])
            print('[Warning] Test set is empty!')
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


class CIFDataPredict(Dataset):
    """
    Dataset for prediction on new CIF files without labels.

    Only requires CIF files and atom_init.json.
    """

    def __init__(
        self,
        cif_dir,
        atom_init_file,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2
    ):
        """
        Initialize prediction dataset.

        Parameters
        ----------
        cif_dir : str
            Directory containing CIF files
        atom_init_file : str
            Path to atom_init.json file
        max_num_nbr : int
            Maximum number of neighbors
        radius : float
            Cutoff radius for neighbor search (Angstroms)
        dmin : float
            Minimum distance for Gaussian expansion
        step : float
            Step size for Gaussian expansion
        """
        self.cif_dir = cif_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius

        assert os.path.exists(cif_dir), f'cif_dir {cif_dir} does not exist!'
        assert os.path.exists(atom_init_file), f'atom_init_file {atom_init_file} does not exist!'

        # Find all CIF files
        self.cif_files = []
        for file in os.listdir(cif_dir):
            if file.endswith('.cif'):
                self.cif_files.append(file[:-4])  # Remove .cif extension

        if len(self.cif_files) == 0:
            raise ValueError(f"No CIF files found in {cif_dir}")

        # Load atom initialization
        self.ari = AtomCustomJSONInitializer(atom_init_file)

        # Initialize Gaussian distance filter
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.cif_files)

    # Removed lru_cache to prevent memory issues with large datasets
    def __getitem__(self, idx):
        """
        Get a single data point.

        Returns
        -------
        (atom_fea, nbr_fea, nbr_fea_idx) : tuple
            Graph representation
        target : torch.Tensor
            Dummy target (zeros)
        cif_id : str
            Crystal identifier
        """
        cif_id = self.cif_files[idx]

        # Load crystal structure
        crystal = Structure.from_file(
            os.path.join(self.cif_dir, cif_id + '.cif')
        )

        # Get atom features
        atom_fea = np.vstack([
            self.ari.get_atom_fea(crystal[i].specie.number)
            for i in range(len(crystal))
        ])
        atom_fea = torch.Tensor(atom_fea)

        # Get all neighbors
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        # Build neighbor feature arrays
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f'{cif_id} does not have enough neighbors. '
                    f'Consider increasing radius.'
                )
                # Pad with zeros
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) +
                    [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr[:self.max_num_nbr]))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                )

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)

        # Apply Gaussian distance filter
        nbr_fea = self.gdf.expand(nbr_fea)

        # Convert to tensors
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        # Return dummy target (0) since we don't have labels
        target = torch.Tensor([0])

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
