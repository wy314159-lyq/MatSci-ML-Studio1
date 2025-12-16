"""
ALIGNN Data Loader
Handle data loading and dataset preparation for ALIGNN
Based on reference_alignn/alignn/data.py and dataset.py
"""

import os
import random
import math
import torch
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
import dgl
from dgl.dataloading import GraphDataLoader

from .graph_builder import ALIGNNGraphBuilder, compute_bond_cosines


def get_train_val_test_split(
    total_size: int,
    split_seed: int = 123,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_val: Optional[int] = None,
    keep_data_order: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    """
    Get train/val/test split indices

    Args:
        total_size: Total number of samples
        split_seed: Random seed
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        n_train: Explicit number of training samples
        n_test: Explicit number of test samples
        n_val: Explicit number of validation samples
        keep_data_order: Keep original order (no shuffle)

    Returns:
        id_train, id_val, id_test: Lists of indices
    """
    # Calculate sizes
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)

    # Check total
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            f"Total samples ({n_train} + {n_val} + {n_test} = {n_train + n_val + n_test}) "
            f"exceeds dataset size ({total_size})"
        )

    # Create indices
    ids = list(np.arange(total_size))

    # Shuffle if requested
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)

    # Split
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test):-n_test] if n_test > 0 else ids[-(n_val + n_test):]
    id_test = ids[-n_test:] if n_test > 0 else []

    return id_train, id_val, id_test


class StructureDataset(torch.utils.data.Dataset):
    """
    Dataset of crystal structures as DGL graphs

    Based on reference_alignn/alignn/graphs.py StructureDataset
    Supports multi-task learning with atomwise, forces, and stress labels
    """

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: List,
        target: str,
        id_tag: str = "id",
        atom_features: str = "cgcnn",
        line_graph: bool = True,
        classification: bool = False,
        dtype: str = "float32",
        target_atomwise: str = "",
        target_grad: str = "",
        target_stress: str = ""
    ):
        """
        Initialize dataset

        Args:
            df: DataFrame with structure info and targets
            graphs: List of DGL graphs
            target: Target property column name (graph-level)
            id_tag: ID column name
            atom_features: Atom feature type
            line_graph: Whether using line graphs
            classification: Whether classification task
            dtype: Data type
            target_atomwise: Atomwise property column (per-atom labels)
            target_grad: Gradient/forces column (per-atom 3D vectors)
            target_stress: Stress tensor column (per-structure 3x3 or 6-element)
        """
        self.df = df
        self.graphs = graphs
        self.target = target
        self.target_atomwise = target_atomwise
        self.target_grad = target_grad
        self.target_stress = target_stress
        self.line_graph = line_graph
        self.classification = classification

        # Get IDs and labels
        self.ids = self.df[id_tag].tolist()
        self.labels = torch.tensor(self.df[target].values).type(torch.get_default_dtype())

        # Get lattices
        try:
            from jarvis.core.atoms import Atoms
            self.lattices = []
            for _, row in df.iterrows():
                atoms = Atoms.from_dict(row["atoms"])
                self.lattices.append(atoms.lattice_mat)
            self.lattices = torch.tensor(np.array(self.lattices)).type(torch.get_default_dtype())
        except:
            # Fallback: create dummy lattices
            self.lattices = torch.eye(3).unsqueeze(0).repeat(len(df), 1, 1).type(torch.get_default_dtype())

        # Load atom features
        features = self._get_attribute_lookup(atom_features)

        # Multi-task labels
        self.labels_atomwise = None
        self.labels_grad = None
        self.labels_stress = None

        # Process atomwise labels (per-atom properties)
        if target_atomwise and target_atomwise != "":
            self.labels_atomwise = []
            for _, row in df.iterrows():
                atomwise_data = np.array(row[target_atomwise])
                self.labels_atomwise.append(
                    torch.tensor(atomwise_data).type(torch.get_default_dtype())
                )

        # Process gradient/forces labels (per-atom 3D vectors)
        if target_grad and target_grad != "":
            self.labels_grad = []
            for _, row in df.iterrows():
                grad_data = np.array(row[target_grad])
                self.labels_grad.append(
                    torch.tensor(grad_data).type(torch.get_default_dtype())
                )

        # Process stress labels (per-structure tensors)
        if target_stress and target_stress != "":
            self.labels_stress = []
            for _, row in df.iterrows():
                self.labels_stress.append(row[target_stress])

        # Process graphs
        for i, g in enumerate(graphs):
            # Load atom features
            z = g.ndata.pop("atom_features")
            g.ndata["atomic_number"] = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.num_nodes() == 1:
                f = f.unsqueeze(0)
            g.ndata["atom_features"] = f

            # Add multi-task labels to graph nodes
            if self.labels_atomwise is not None:
                g.ndata[target_atomwise] = self.labels_atomwise[i]
            if self.labels_grad is not None:
                g.ndata[target_grad] = self.labels_grad[i]
            if self.labels_stress is not None:
                # Stress is per-structure, replicate for all atoms
                num_atoms = g.num_nodes()
                stress_tensor = torch.tensor([self.labels_stress[i]] * num_atoms).type(
                    torch.get_default_dtype()
                )
                g.ndata[target_stress] = stress_tensor

        # Build line graphs if needed
        if line_graph:
            self.line_graphs = []
            print("Building line graphs...")
            for g in tqdm(graphs):
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)
                self.line_graphs.append(lg)

        # Convert to long for classification
        if classification:
            self.labels = self.labels.view(-1).long()

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build lookup array indexed by atomic number"""
        try:
            from jarvis.core.specie import chem_data, get_node_attributes

            max_z = max(v["Z"] for v in chem_data.values())
            template = get_node_attributes("C", atom_features)
            features = np.zeros((1 + max_z, len(template)))

            for element, v in chem_data.items():
                z = v["Z"]
                x = get_node_attributes(element, atom_features)
                if x is not None:
                    features[z, :] = x

            return features
        except:
            # Fallback for cgcnn: one-hot encoding
            if atom_features == "cgcnn":
                # Return 92-dimensional one-hot (Z from 1 to 92, indexed by Z-1)
                # features[z] returns one-hot for atomic number z (1-indexed)
                features = np.zeros((93, 92))  # 93 rows (0-92), 92 columns
                for z in range(1, 93):
                    features[z, z-1] = 1.0  # Z=1 → [1,0,0,...], Z=2 → [0,1,0,...], ...
                return features
            else:
                raise NotImplementedError(f"Feature type {atom_features} requires jarvis")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item by index"""
        g = self.graphs[idx]
        label = self.labels[idx]
        lattice = self.lattices[idx]

        if self.line_graph:
            return g, self.line_graphs[idx], lattice, label
        else:
            return g, lattice, label

    @staticmethod
    def collate_line_graph(samples):
        """Collate function for line graph batches"""
        graphs, line_graphs, lattices, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)

        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(lattices), torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.stack(lattices), torch.tensor(labels)

    @staticmethod
    def collate(samples):
        """Collate function for regular batches"""
        graphs, lattices, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(lattices), torch.tensor(labels)


class ALIGNNDataLoader:
    """
    ALIGNN Data Loader

    High-level interface for loading and preparing data for ALIGNN
    """

    def __init__(
        self,
        data_path: str,
        target: str = "target",
        id_tag: str = "id",
        atom_features: str = "cgcnn",
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        use_canonize: bool = True,
        compute_line_graph: bool = True,
        neighbor_strategy: str = "k-nearest",
        dtype: str = "float32"
    ):
        """
        Initialize data loader

        Args:
            data_path: Path to CSV/Excel file with structure data
            target: Target property column
            id_tag: ID column
            atom_features: Atom feature type
            cutoff: Cutoff radius
            max_neighbors: Max neighbors
            use_canonize: Canonize edges
            compute_line_graph: Compute line graph
            neighbor_strategy: Neighbor strategy
            dtype: Data type
        """
        self.data_path = data_path
        self.target = target
        self.id_tag = id_tag
        self.atom_features = atom_features
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_canonize = use_canonize
        self.compute_line_graph = compute_line_graph
        self.neighbor_strategy = neighbor_strategy
        self.dtype = dtype

        # Initialize graph builder
        self.graph_builder = ALIGNNGraphBuilder(
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            atom_features=atom_features,
            compute_line_graph=False,  # We'll handle this in dataset
            use_canonize=use_canonize,
            neighbor_strategy=neighbor_strategy,
            dtype=dtype
        )

        self.df = None
        self.dataset = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from file

        Returns:
            DataFrame with structure data
        """
        # Load file
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")

        print(f"Loaded {len(self.df)} structures from {self.data_path}")

        # Validate columns
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data")
        if self.id_tag not in self.df.columns:
            raise ValueError(f"ID column '{self.id_tag}' not found in data")
        if "atoms" not in self.df.columns:
            raise ValueError("'atoms' column not found in data")

        return self.df

    def build_graphs(self) -> List:
        """
        Build DGL graphs from structures

        Returns:
            List of graphs
        """
        if self.df is None:
            raise ValueError("Must call load_data() first")

        print("Converting structures to graphs...")
        graphs = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                # Get atoms
                atoms_dict = row["atoms"]

                # Import Atoms class
                try:
                    from jarvis.core.atoms import Atoms
                    atoms = Atoms.from_dict(atoms_dict)
                except ImportError:
                    raise ImportError("jarvis-tools is required for structure handling")

                # Build graph
                g = self.graph_builder.atoms_to_graph(atoms, id=row[self.id_tag])
                graphs.append(g)

            except Exception as e:
                print(f"Error processing structure {row[self.id_tag]}: {e}")
                continue

        print(f"Successfully built {len(graphs)} graphs")
        return graphs

    def create_dataset(
        self,
        graphs: Optional[List] = None,
        classification: bool = False
    ) -> StructureDataset:
        """
        Create PyTorch dataset

        Args:
            graphs: Pre-built graphs (optional)
            classification: Classification task

        Returns:
            StructureDataset instance
        """
        if self.df is None:
            raise ValueError("Must call load_data() first")

        if graphs is None:
            graphs = self.build_graphs()

        self.dataset = StructureDataset(
            df=self.df,
            graphs=graphs,
            target=self.target,
            id_tag=self.id_tag,
            atom_features=self.atom_features,
            line_graph=self.compute_line_graph,
            classification=classification,
            dtype=self.dtype
        )

        return self.dataset

    def get_data_loaders(
        self,
        batch_size: int = 64,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 123,
        num_workers: int = 4,
        pin_memory: bool = False,
        classification: bool = False,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        n_test: Optional[int] = None,
        keep_data_order: bool = False
    ) -> Tuple:
        """
        Get train/val/test data loaders

        Args:
            batch_size: Batch size
            train_ratio: Training ratio
            val_ratio: Validation ratio
            test_ratio: Test ratio
            split_seed: Random seed
            num_workers: Number of workers
            pin_memory: Pin memory
            classification: Classification task
            n_train: Number of training samples
            n_val: Number of validation samples
            n_test: Number of test samples
            keep_data_order: Keep data order

        Returns:
            (train_loader, val_loader, test_loader, dataset)
        """
        # Create dataset if not exists
        if self.dataset is None:
            self.create_dataset(classification=classification)

        # Split indices
        id_train, id_val, id_test = get_train_val_test_split(
            total_size=len(self.dataset),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            keep_data_order=keep_data_order
        )

        # Create subsets
        train_dataset = torch.utils.data.Subset(self.dataset, id_train)
        val_dataset = torch.utils.data.Subset(self.dataset, id_val)
        test_dataset = torch.utils.data.Subset(self.dataset, id_test)

        # Collate function
        collate_fn = (
            self.dataset.collate_line_graph if self.compute_line_graph
            else self.dataset.collate
        )

        # Create loaders
        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_loader = GraphDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = GraphDataLoader(
            test_dataset,
            batch_size=1,  # Test with batch size 1
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        print(f"Data loaders created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

        return train_loader, val_loader, test_loader, self.dataset

    def save_to_lmdb(
        self,
        output_path: str,
        graphs: Optional[List] = None,
        map_size: int = 1099511627776
    ):
        """
        Save dataset to LMDB for efficient loading.

        Args:
            output_path: Output directory for LMDB
            graphs: Pre-built graphs (optional, will build if not provided)
            map_size: Maximum LMDB size (default 1TB)
        """
        from .lmdb_dataset import create_lmdb_dataset

        if graphs is None:
            graphs = self.build_graphs()

        if self.dataset is None:
            self.dataset = self.create_dataset(graphs=graphs)

        print(f"Saving dataset to LMDB at {output_path}...")

        # Extract data
        labels = self.dataset.labels
        lattices = self.dataset.lattices
        line_graphs = self.dataset.line_graphs if self.compute_line_graph else None

        # Create LMDB
        create_lmdb_dataset(
            graphs=graphs,
            labels=labels,
            lattices=lattices,
            lmdb_path=output_path,
            line_graphs=line_graphs,
            map_size=map_size
        )

        print(f"LMDB dataset saved successfully")

    @staticmethod
    def load_from_lmdb(
        lmdb_path: str,
        batch_size: int = 64,
        line_graph: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 123
    ):
        """
        Load dataset from LMDB.

        Args:
            lmdb_path: Path to LMDB directory
            batch_size: Batch size
            line_graph: Whether to use line graphs
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for splitting

        Returns:
            train_loader, val_loader, test_loader
        """
        from .lmdb_dataset import get_lmdb_loaders

        return get_lmdb_loaders(
            lmdb_path=lmdb_path,
            batch_size=batch_size,
            line_graph=line_graph,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )

