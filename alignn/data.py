"""
ALIGNN Data Module
Based on CGCNN data.py design pattern with enhanced usability

Features:
- Automatic CIF file discovery
- Flexible property file formats (CSV, Excel)
- Automatic train/val/test splitting
- Data validation and preprocessing
- Support for jarvis Atoms format
- Line graph generation
"""

import os
import csv
import json
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
from tqdm import tqdm

from .graph_builder import ALIGNNGraphBuilder, compute_bond_cosines


# =============================================================================
# Atom Feature Initialization
# =============================================================================

class AtomInitializer:
    """
    Base class for initializing atom feature vectors.
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        """Get feature vector for atom type (atomic number)."""
        if atom_type not in self.atom_types:
            supported = sorted(list(self.atom_types))[:20]
            raise ValueError(
                f"Element with atomic number {atom_type} is NOT supported!\n\n"
                f"Supported elements (first 20): {supported}...\n"
                f"Total supported elements: {len(self.atom_types)}\n\n"
                f"Please check your structures or update atom features."
            )
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        """Load embedding from state dict."""
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())

    def state_dict(self):
        """Return embedding state dict."""
        return self._embedding


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom features from JSON file.
    The JSON file should map element numbers to feature vectors.
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)

        elem_embedding = {
            int(key): value for key, value in elem_embedding.items()
        }
        atom_types = set(elem_embedding.keys())

        super().__init__(atom_types)

        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class AtomJarvisInitializer(AtomInitializer):
    """
    Initialize atom features using jarvis-tools specie data.
    Supports: 'cgcnn', 'basic', 'atomic_number', 'cfid'
    """

    def __init__(self, atom_features: str = "cgcnn"):
        self.atom_features = atom_features

        try:
            from jarvis.core.specie import chem_data, get_node_attributes
            self.chem_data = chem_data
            self.get_node_attributes = get_node_attributes

            # Build embedding for all known elements
            atom_types = set()
            for element, data in chem_data.items():
                z = data["Z"]
                atom_types.add(z)

            super().__init__(atom_types)

            for element, data in chem_data.items():
                z = data["Z"]
                feat = get_node_attributes(element, atom_features=atom_features)
                if feat is not None:
                    self._embedding[z] = np.array(feat, dtype=float)

        except ImportError:
            # Fallback: cgcnn one-hot encoding
            if atom_features == "cgcnn":
                atom_types = set(range(1, 101))  # Z=1 to Z=100
                super().__init__(atom_types)

                for z in range(1, 101):
                    feat = np.zeros(92, dtype=float)
                    if z <= 92:
                        feat[z - 1] = 1.0
                    self._embedding[z] = feat
            else:
                raise ImportError(
                    f"jarvis-tools required for atom_features='{atom_features}'. "
                    f"Install with: pip install jarvis-tools"
                )


# =============================================================================
# CIF Dataset (CGCNN-style interface)
# =============================================================================

class CIFData(Dataset):
    """
    Dataset for crystal structures stored as CIF files.
    Compatible with CGCNN-style directory structure.

    Expected directory structure:
        root_dir/
            id_prop.csv       # CSV with: cif_id, target
            atom_init.json    # (optional) atom feature embeddings
            *.cif             # CIF structure files
    """

    def __init__(
        self,
        root_dir: str,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        atom_features: str = "cgcnn",
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        neighbor_strategy: str = "k-nearest",
        random_seed: int = 123,
        dtype: str = "float32",
        keep_data_order: bool = True,
        classification_threshold: Optional[float] = None,
        target_multiplication_factor: Optional[float] = None
    ):
        """
        Initialize CIF dataset.

        Parameters
        ----------
        root_dir : str
            Directory containing CIF files and id_prop.csv
        max_num_nbr : int
            Maximum number of neighbors (for k-nearest)
        radius : float
            Cutoff radius for neighbor search (Angstroms)
        atom_features : str
            Atom feature type: 'cgcnn', 'basic', 'atomic_number', 'cfid'
        compute_line_graph : bool
            Whether to compute line graphs for angles
        use_canonize : bool
            Whether to canonize edges
        neighbor_strategy : str
            'k-nearest', 'radius_graph', 'voronoi', or 'radius_graph_jarvis'
        random_seed : int
            Random seed for shuffling
        dtype : str
            Data type ('float32', 'float64')
        keep_data_order : bool
            If True, keep original data order (no shuffle). If False, shuffle data.
        classification_threshold : float, optional
            If set, convert continuous targets to binary classification:
            target = 1 if value > threshold else 0
            Aligned with official ALIGNN implementation.
        target_multiplication_factor : float, optional
            Multiply target values by this factor before use.
            Aligned with official ALIGNN implementation.
        """
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.atom_features = atom_features
        self.compute_line_graph = compute_line_graph
        self.neighbor_strategy = neighbor_strategy
        self.dtype = dtype
        self.keep_data_order = keep_data_order
        self.classification_threshold = classification_threshold
        self.target_multiplication_factor = target_multiplication_factor

        assert os.path.exists(root_dir), f'root_dir {root_dir} does not exist!'

        # Load property data
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'

        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Shuffle data only if keep_data_order is False
        if not keep_data_order:
            random.seed(random_seed)
            random.shuffle(self.id_prop_data)

        # Initialize atom features
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        if os.path.exists(atom_init_file):
            self.ari = AtomCustomJSONInitializer(atom_init_file)
        else:
            self.ari = AtomJarvisInitializer(atom_features)

        # Initialize graph builder
        self.graph_builder = ALIGNNGraphBuilder(
            cutoff=radius,
            max_neighbors=max_num_nbr,
            atom_features=atom_features,
            compute_line_graph=False,
            use_canonize=use_canonize,
            neighbor_strategy=neighbor_strategy,
            dtype=dtype
        )

        # Set torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        self.torch_dtype = dtype_map.get(dtype, torch.float32)

        # Cache for graphs and line graphs
        self._graph_cache = {}
        self._line_graph_cache = {}

    def __len__(self):
        return len(self.id_prop_data)

    def _load_structure(self, cif_id: str):
        """Load crystal structure from CIF file."""
        cif_path = os.path.join(self.root_dir, cif_id + '.cif')

        # Try jarvis Atoms first
        try:
            from jarvis.core.atoms import Atoms
            atoms = Atoms.from_cif(cif_path)
            return atoms, "jarvis"
        except:
            pass

        # Try pymatgen Structure
        try:
            from pymatgen.core.structure import Structure
            structure = Structure.from_file(cif_path)
            return structure, "pymatgen"
        except:
            pass

        raise ValueError(f"Cannot load structure from {cif_path}")

    def __getitem__(self, idx):
        """
        Get a single data point.

        Returns
        -------
        (graph, line_graph, lattice) or (graph, lattice) : tuple
            Graph representation
        target : torch.Tensor
            Target property value
        cif_id : str
            Crystal identifier
        """
        cif_id, target = self.id_prop_data[idx]

        # Check cache
        if idx in self._graph_cache:
            g = self._graph_cache[idx]
            lg = self._line_graph_cache.get(idx)
        else:
            # Load structure
            structure, struct_type = self._load_structure(cif_id)

            # Build graph
            if struct_type == "jarvis":
                g = self.graph_builder.atoms_to_graph(structure, id=cif_id)
                lattice = torch.tensor(structure.lattice_mat).type(self.torch_dtype)
            else:
                # Convert pymatgen to jarvis
                try:
                    from jarvis.core.atoms import Atoms
                    atoms = Atoms.from_dict({
                        "lattice_mat": structure.lattice.matrix.tolist(),
                        "coords": structure.frac_coords.tolist(),
                        "elements": [str(s.specie) for s in structure.sites],
                        "cartesian": False
                    })
                    g = self.graph_builder.atoms_to_graph(atoms, id=cif_id)
                    lattice = torch.tensor(atoms.lattice_mat).type(self.torch_dtype)
                except:
                    raise ValueError(f"Cannot process structure {cif_id}")

            # Update atom features using initializer
            # Check if atom_features already contains full vectors (from radius_graph_jarvis)
            # or atomic numbers (from k-nearest/radius_graph)
            current_features = g.ndata["atom_features"]
            if current_features.dim() == 1:
                # 1D tensor: contains atomic numbers, need to convert to features
                z = current_features.long()
                atom_fea = []
                for atomic_num in z.numpy():
                    feat = self.ari.get_atom_fea(int(atomic_num))
                    atom_fea.append(feat)
                g.ndata["atom_features"] = torch.tensor(
                    np.array(atom_fea), dtype=self.torch_dtype
                )
            else:
                # 2D tensor: already contains feature vectors (from radius_graph_jarvis)
                # Just ensure correct dtype
                g.ndata["atom_features"] = current_features.type(self.torch_dtype)

            # Build line graph if needed
            if self.compute_line_graph:
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)
                self._line_graph_cache[idx] = lg
            else:
                lg = None

            # Cache
            self._graph_cache[idx] = g

        # Get lattice from graph
        if hasattr(g, 'ndata') and 'frac_coords' in g.ndata:
            try:
                structure, _ = self._load_structure(cif_id)
                if hasattr(structure, 'lattice_mat'):
                    lattice = torch.tensor(structure.lattice_mat).type(self.torch_dtype)
                else:
                    lattice = torch.tensor(structure.lattice.matrix).type(self.torch_dtype)
            except:
                lattice = torch.eye(3).type(self.torch_dtype)
        else:
            try:
                structure, _ = self._load_structure(cif_id)
                if hasattr(structure, 'lattice_mat'):
                    lattice = torch.tensor(structure.lattice_mat).type(self.torch_dtype)
                else:
                    lattice = torch.tensor(structure.lattice.matrix).type(self.torch_dtype)
            except:
                lattice = torch.eye(3).type(self.torch_dtype)

        # Process target value (aligned with official ALIGNN)
        target_val = float(target)

        # Apply multiplication factor if specified
        if self.target_multiplication_factor is not None:
            target_val = target_val * self.target_multiplication_factor

        # Apply classification threshold if specified
        if self.classification_threshold is not None:
            # Convert to binary classification: 1 if > threshold, else 0
            target_val = 1 if target_val > self.classification_threshold else 0
            target = torch.tensor([target_val], dtype=torch.long)
        else:
            target = torch.tensor([target_val], dtype=self.torch_dtype)

        if self.compute_line_graph:
            return (g, lg, lattice), target, cif_id
        else:
            return (g, lattice), target, cif_id


class CIFDataPredict(Dataset):
    """
    Dataset for prediction on new CIF files without labels.
    Only requires CIF files directory.
    """

    def __init__(
        self,
        cif_dir: str,
        atom_init_file: Optional[str] = None,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        atom_features: str = "cgcnn",
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        neighbor_strategy: str = "k-nearest",
        dtype: str = "float32"
    ):
        """
        Initialize prediction dataset.

        Parameters
        ----------
        cif_dir : str
            Directory containing CIF files
        atom_init_file : str, optional
            Path to atom_init.json (uses jarvis if not provided)
        max_num_nbr : int
            Maximum number of neighbors
        radius : float
            Cutoff radius (Angstroms)
        atom_features : str
            Atom feature type
        compute_line_graph : bool
            Whether to compute line graphs
        use_canonize : bool
            Whether to canonize edges
        neighbor_strategy : str
            Neighbor strategy
        dtype : str
            Data type
        """
        self.cif_dir = cif_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.atom_features = atom_features
        self.compute_line_graph = compute_line_graph
        self.dtype = dtype

        assert os.path.exists(cif_dir), f'cif_dir {cif_dir} does not exist!'

        # Find all CIF files
        self.cif_files = []
        for file in os.listdir(cif_dir):
            if file.endswith('.cif'):
                self.cif_files.append(file[:-4])

        if len(self.cif_files) == 0:
            raise ValueError(f"No CIF files found in {cif_dir}")

        # Initialize atom features
        if atom_init_file and os.path.exists(atom_init_file):
            self.ari = AtomCustomJSONInitializer(atom_init_file)
        else:
            self.ari = AtomJarvisInitializer(atom_features)

        # Initialize graph builder
        self.graph_builder = ALIGNNGraphBuilder(
            cutoff=radius,
            max_neighbors=max_num_nbr,
            atom_features=atom_features,
            compute_line_graph=False,
            use_canonize=use_canonize,
            neighbor_strategy=neighbor_strategy,
            dtype=dtype
        )

        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        self.torch_dtype = dtype_map.get(dtype, torch.float32)

    def __len__(self):
        return len(self.cif_files)

    def __getitem__(self, idx):
        """Get a single data point (no label)."""
        cif_id = self.cif_files[idx]
        cif_path = os.path.join(self.cif_dir, cif_id + '.cif')

        # Load structure
        try:
            from jarvis.core.atoms import Atoms
            atoms = Atoms.from_cif(cif_path)
            lattice = torch.tensor(atoms.lattice_mat).type(self.torch_dtype)
        except:
            from pymatgen.core.structure import Structure
            structure = Structure.from_file(cif_path)
            from jarvis.core.atoms import Atoms
            atoms = Atoms.from_dict({
                "lattice_mat": structure.lattice.matrix.tolist(),
                "coords": structure.frac_coords.tolist(),
                "elements": [str(s.specie) for s in structure.sites],
                "cartesian": False
            })
            lattice = torch.tensor(atoms.lattice_mat).type(self.torch_dtype)

        # Build graph
        g = self.graph_builder.atoms_to_graph(atoms, id=cif_id)

        # Update atom features
        # Check if atom_features already contains full vectors (from radius_graph_jarvis)
        # or atomic numbers (from k-nearest/radius_graph)
        current_features = g.ndata["atom_features"]
        if current_features.dim() == 1:
            # 1D tensor: contains atomic numbers, need to convert to features
            z = current_features.long()
            atom_fea = []
            for atomic_num in z.numpy():
                feat = self.ari.get_atom_fea(int(atomic_num))
                atom_fea.append(feat)
            g.ndata["atom_features"] = torch.tensor(
                np.array(atom_fea), dtype=self.torch_dtype
            )
        else:
            # 2D tensor: already contains feature vectors (from radius_graph_jarvis)
            # Just ensure correct dtype
            g.ndata["atom_features"] = current_features.type(self.torch_dtype)

        # Build line graph if needed
        if self.compute_line_graph:
            lg = g.line_graph(shared=True)
            lg.apply_edges(compute_bond_cosines)
            return (g, lg, lattice), torch.tensor([0.0]), cif_id
        else:
            return (g, lattice), torch.tensor([0.0]), cif_id


# =============================================================================
# Collate Functions
# =============================================================================

def collate_line_graph(samples):
    """
    Collate function for batching ALIGNN data with line graphs.

    Parameters
    ----------
    samples : list
        List of ((g, lg, lattice), target, cif_id) tuples

    Returns
    -------
    (batched_graph, batched_line_graph, lattices) : tuple
    targets : torch.Tensor
    cif_ids : list
    """
    graphs, targets, cif_ids = [], [], []
    line_graphs, lattices = [], []

    for (g, lg, lat), target, cif_id in samples:
        graphs.append(g)
        line_graphs.append(lg)
        lattices.append(lat)
        targets.append(target)
        cif_ids.append(cif_id)

    batched_graph = dgl.batch(graphs)
    batched_line_graph = dgl.batch(line_graphs)
    batched_lattices = torch.stack(lattices)
    batched_targets = torch.stack(targets)

    return (batched_graph, batched_line_graph, batched_lattices), batched_targets, cif_ids


def collate_no_line_graph(samples):
    """
    Collate function for batching ALIGNN data without line graphs.

    Parameters
    ----------
    samples : list
        List of ((g, lattice), target, cif_id) tuples

    Returns
    -------
    (batched_graph, lattices) : tuple
    targets : torch.Tensor
    cif_ids : list
    """
    graphs, targets, cif_ids = [], [], []
    lattices = []

    for (g, lat), target, cif_id in samples:
        graphs.append(g)
        lattices.append(lat)
        targets.append(target)
        cif_ids.append(cif_id)

    batched_graph = dgl.batch(graphs)
    batched_lattices = torch.stack(lattices)
    batched_targets = torch.stack(targets)

    return (batched_graph, batched_lattices), batched_targets, cif_ids


# =============================================================================
# Data Loader Factory
# =============================================================================

def get_train_val_test_loader(
    dataset: Dataset,
    collate_fn=None,
    batch_size: int = 64,
    train_ratio: Optional[float] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    return_test: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    random_seed: Optional[int] = 123,
    shuffle: bool = True,
    output_dir: Optional[str] = None,
    split_file: Optional[str] = None,
    save_split: bool = True,
    keep_data_order: bool = True,
    **kwargs
):
    """
    Create train/val/test data loaders.

    Compatible with CGCNN implementation style.
    Supports both ratio-based and absolute size-based splitting.
    Aligned with official ALIGNN: supports saving/loading ids_train_val_test.json.

    Parameters
    ----------
    dataset : Dataset
        Full dataset to split
    collate_fn : callable, optional
        Collate function (auto-detected if None)
    batch_size : int
        Batch size (default: 64)
    train_ratio : float, optional
        Fraction for training
    val_ratio : float
        Fraction for validation (default: 0.1)
    test_ratio : float
        Fraction for testing (default: 0.1)
    return_test : bool
        Whether to return test loader (default: True)
    num_workers : int
        Number of data loading workers (default: 0)
    pin_memory : bool
        Whether to pin memory (default: False)
    train_size : int, optional
        Absolute number of training samples
    val_size : int, optional
        Absolute number of validation samples
    test_size : int, optional
        Absolute number of test samples
    random_seed : int, optional
        Random seed for reproducible splits
    shuffle : bool
        Whether to shuffle indices before splitting (default: True)
    output_dir : str, optional
        Directory to save ids_train_val_test.json (default: None)
    split_file : str, optional
        Path to existing ids_train_val_test.json to load splits from
    save_split : bool
        Whether to save split IDs to JSON (default: True)
    keep_data_order : bool
        If True, keep original data order (no shuffle). Overrides shuffle param.
        Aligned with official ALIGNN (default: True)

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader (if return_test=True)
    """
    # Auto-detect collate function
    if collate_fn is None:
        if hasattr(dataset, 'compute_line_graph') and dataset.compute_line_graph:
            collate_fn = collate_line_graph
        else:
            collate_fn = collate_no_line_graph

    total_size = len(dataset)

    # Get IDs from dataset (for saving to JSON)
    def get_id_from_dataset(idx):
        """Get ID for a given index from dataset."""
        if hasattr(dataset, 'id_prop_data'):
            # CIFData format: id_prop_data is list of [cif_id, target]
            return dataset.id_prop_data[idx][0]
        elif hasattr(dataset, 'df') and hasattr(dataset, 'id_tag'):
            # JarvisData format
            return str(dataset.df.iloc[idx].get(dataset.id_tag, idx))
        else:
            return str(idx)

    # Check if loading from existing split file
    if split_file and os.path.exists(split_file):
        print(f'[Data Split] Loading split from: {split_file}')
        with open(split_file, 'r') as f:
            split_data = json.load(f)

        # Get ID to index mapping
        id_to_idx = {}
        for idx in range(total_size):
            id_val = get_id_from_dataset(idx)
            id_to_idx[str(id_val)] = idx

        # Convert IDs to indices
        train_indices = [id_to_idx[str(id_val)] for id_val in split_data.get('id_train', []) if str(id_val) in id_to_idx]
        val_indices = [id_to_idx[str(id_val)] for id_val in split_data.get('id_val', []) if str(id_val) in id_to_idx]
        test_indices = [id_to_idx[str(id_val)] for id_val in split_data.get('id_test', []) if str(id_val) in id_to_idx]

        print(f'[Data Split] Loaded - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}')

    else:
        # Calculate sizes
        if train_size is None:
            if train_ratio is None:
                assert val_ratio + test_ratio < 1
                train_ratio = 1 - val_ratio - test_ratio
                print(f'[Info] train_ratio set to {train_ratio:.2f}')
            train_size = int(train_ratio * total_size)

        if test_size is None:
            test_size = int(test_ratio * total_size)

        if val_size is None:
            val_size = int(val_ratio * total_size)

        # Ensure minimum sizes
        if val_ratio > 0 and val_size == 0 and total_size >= 3:
            val_size = 1
        if test_ratio > 0 and test_size == 0 and total_size >= 3:
            test_size = 1

        # Adjust if exceeds total
        if train_size + val_size + test_size > total_size:
            train_size = total_size - val_size - test_size
            if train_size < 1:
                train_size = max(1, total_size - 2)
                val_size = max(0, (total_size - train_size) // 2)
                test_size = total_size - train_size - val_size

        print(f'[Data Split] Total: {total_size}, Train: {train_size}, '
              f'Val: {val_size}, Test: {test_size}')

        # Create indices
        indices = list(range(total_size))

        # Shuffle only if keep_data_order is False (aligned with official ALIGNN)
        if not keep_data_order and shuffle:
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(indices)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]

        # Save split to JSON (aligned with official ALIGNN)
        if save_split and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            split_path = os.path.join(output_dir, 'ids_train_val_test.json')

            # Get IDs for each split
            id_train = [get_id_from_dataset(idx) for idx in train_indices]
            id_val = [get_id_from_dataset(idx) for idx in val_indices]
            id_test = [get_id_from_dataset(idx) for idx in test_indices]

            split_data = {
                'id_train': id_train,
                'id_val': id_val,
                'id_test': id_test
            }

            with open(split_path, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f'[Data Split] Saved split IDs to: {split_path}')

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices) if len(val_indices) > 0 else SubsetRandomSampler([])

    # Create loaders
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
        test_sampler = SubsetRandomSampler(test_indices) if len(test_indices) > 0 else SubsetRandomSampler([])
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


# =============================================================================
# Jarvis Dataset (for jarvis-tools format)
# =============================================================================

class JarvisData(Dataset):
    """
    Dataset for jarvis-tools format data.
    Supports loading from DataFrame with 'atoms' column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        id_tag: str = "jid",
        max_num_nbr: int = 12,
        radius: float = 8.0,
        atom_features: str = "cgcnn",
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        neighbor_strategy: str = "k-nearest",
        dtype: str = "float32",
        precompute_graphs: bool = True,
        classification_threshold: Optional[float] = None,
        target_multiplication_factor: Optional[float] = None
    ):
        """
        Initialize Jarvis dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'atoms' column (jarvis Atoms dicts) and target column
        target : str
            Target property column name
        id_tag : str
            ID column name
        max_num_nbr : int
            Maximum number of neighbors
        radius : float
            Cutoff radius (Angstroms)
        atom_features : str
            Atom feature type
        compute_line_graph : bool
            Whether to compute line graphs
        use_canonize : bool
            Whether to canonize edges
        neighbor_strategy : str
            Neighbor strategy
        dtype : str
            Data type
        precompute_graphs : bool
            Whether to precompute all graphs
        classification_threshold : float, optional
            If set, convert continuous targets to binary classification.
            Aligned with official ALIGNN implementation.
        target_multiplication_factor : float, optional
            Multiply target values by this factor before use.
            Aligned with official ALIGNN implementation.
        """
        self.df = df.reset_index(drop=True)
        self.target = target
        self.id_tag = id_tag
        self.compute_line_graph = compute_line_graph
        self.atom_features = atom_features
        self.classification_threshold = classification_threshold
        self.target_multiplication_factor = target_multiplication_factor

        # Validate columns
        assert target in df.columns, f"Target '{target}' not in DataFrame"
        assert "atoms" in df.columns, "'atoms' column required"

        # Initialize atom features
        self.ari = AtomJarvisInitializer(atom_features)

        # Initialize graph builder
        self.graph_builder = ALIGNNGraphBuilder(
            cutoff=radius,
            max_neighbors=max_num_nbr,
            atom_features=atom_features,
            compute_line_graph=False,
            use_canonize=use_canonize,
            neighbor_strategy=neighbor_strategy,
            dtype=dtype
        )

        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        self.torch_dtype = dtype_map.get(dtype, torch.float32)

        # Precompute graphs if requested
        self.graphs = None
        self.line_graphs = None
        self.lattices = None

        if precompute_graphs:
            self._precompute_graphs()

    def _precompute_graphs(self):
        """Precompute all graphs."""
        from jarvis.core.atoms import Atoms

        print("Precomputing graphs...")
        self.graphs = []
        self.line_graphs = []
        self.lattices = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            atoms = Atoms.from_dict(row["atoms"])
            lattice = torch.tensor(atoms.lattice_mat).type(self.torch_dtype)

            # Build graph
            g = self.graph_builder.atoms_to_graph(
                atoms,
                id=row.get(self.id_tag, str(idx))
            )

            # Update atom features
            # Check if atom_features already contains full vectors (from radius_graph_jarvis)
            # or atomic numbers (from k-nearest/radius_graph)
            current_features = g.ndata["atom_features"]
            if current_features.dim() == 1:
                # 1D tensor: contains atomic numbers, need to convert to features
                z = current_features.long()
                atom_fea = []
                for atomic_num in z.numpy():
                    feat = self.ari.get_atom_fea(int(atomic_num))
                    atom_fea.append(feat)
                g.ndata["atom_features"] = torch.tensor(
                    np.array(atom_fea), dtype=self.torch_dtype
                )
            else:
                # 2D tensor: already contains feature vectors (from radius_graph_jarvis)
                g.ndata["atom_features"] = current_features.type(self.torch_dtype)

            self.graphs.append(g)
            self.lattices.append(lattice)

            # Build line graph
            if self.compute_line_graph:
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)
                self.line_graphs.append(lg)

        print(f"Precomputed {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.df)

    def _process_target(self, target_val):
        """Process target value with multiplication factor and classification threshold."""
        # Apply multiplication factor if specified
        if self.target_multiplication_factor is not None:
            target_val = target_val * self.target_multiplication_factor

        # Apply classification threshold if specified
        if self.classification_threshold is not None:
            # Convert to binary classification: 1 if > threshold, else 0
            target_val = 1 if target_val > self.classification_threshold else 0
            return torch.tensor([target_val], dtype=torch.long)
        else:
            return torch.tensor([target_val], dtype=self.torch_dtype)

    def __getitem__(self, idx):
        """Get item by index."""
        row = self.df.iloc[idx]
        target = self._process_target(float(row[self.target]))
        cif_id = str(row.get(self.id_tag, idx))

        if self.graphs is not None:
            g = self.graphs[idx]
            lattice = self.lattices[idx]
            if self.compute_line_graph:
                lg = self.line_graphs[idx]
                return (g, lg, lattice), target, cif_id
            else:
                return (g, lattice), target, cif_id
        else:
            # Compute on-the-fly
            from jarvis.core.atoms import Atoms
            atoms = Atoms.from_dict(row["atoms"])
            lattice = torch.tensor(atoms.lattice_mat).type(self.torch_dtype)

            g = self.graph_builder.atoms_to_graph(atoms, id=cif_id)

            # Update atom features
            # Check if atom_features already contains full vectors (from radius_graph_jarvis)
            # or atomic numbers (from k-nearest/radius_graph)
            current_features = g.ndata["atom_features"]
            if current_features.dim() == 1:
                # 1D tensor: contains atomic numbers, need to convert to features
                z = current_features.long()
                atom_fea = []
                for atomic_num in z.numpy():
                    feat = self.ari.get_atom_fea(int(atomic_num))
                    atom_fea.append(feat)
                g.ndata["atom_features"] = torch.tensor(
                    np.array(atom_fea), dtype=self.torch_dtype
                )
            else:
                # 2D tensor: already contains feature vectors (from radius_graph_jarvis)
                g.ndata["atom_features"] = current_features.type(self.torch_dtype)

            if self.compute_line_graph:
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)
                return (g, lg, lattice), target, cif_id
            else:
                return (g, lattice), target, cif_id


# =============================================================================
# Convenience Functions
# =============================================================================

def load_dataset(
    data_path: str,
    target: str = "target",
    format: str = "auto",
    **kwargs
) -> Dataset:
    """
    Load dataset from file.

    Parameters
    ----------
    data_path : str
        Path to data file or directory
    target : str
        Target property column/key
    format : str
        Data format: 'auto', 'cif', 'jarvis', 'csv'
    **kwargs
        Additional arguments for dataset constructor

    Returns
    -------
    dataset : Dataset
    """
    if format == "auto":
        if os.path.isdir(data_path):
            if os.path.exists(os.path.join(data_path, "id_prop.csv")):
                format = "cif"
            else:
                format = "cif_predict"
        elif data_path.endswith(".csv"):
            format = "csv"
        elif data_path.endswith(".json"):
            format = "jarvis"
        else:
            raise ValueError(f"Cannot auto-detect format for {data_path}")

    if format == "cif":
        return CIFData(data_path, **kwargs)
    elif format == "cif_predict":
        return CIFDataPredict(data_path, **kwargs)
    elif format == "csv":
        df = pd.read_csv(data_path)
        return JarvisData(df, target=target, **kwargs)
    elif format == "jarvis":
        df = pd.read_json(data_path)
        return JarvisData(df, target=target, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")
