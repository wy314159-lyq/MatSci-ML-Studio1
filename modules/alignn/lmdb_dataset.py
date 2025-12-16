"""
ALIGNN LMDB Dataset

Efficient LMDB-based dataset for large-scale training
Based on reference_alignn/alignn/lmdb_dataset.py
"""

import os
import lmdb
import pickle as pk
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import dgl
from tqdm import tqdm
import numpy as np


def prepare_line_graph_batch(
    batch: Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """
    Send line graph batch to device.

    Args:
        batch: Nested tuple (g, lg, lattice, label)
        device: Target device
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Batch on target device
    """
    g, lg, lattice, label = batch

    return (
        g.to(device, non_blocking=non_blocking),
        lg.to(device, non_blocking=non_blocking),
        lattice.to(device, non_blocking=non_blocking),
        label.to(device, non_blocking=non_blocking),
    )


def prepare_dgl_batch(
    batch: Tuple[dgl.DGLGraph, torch.Tensor],
    device=None,
    non_blocking=False,
):
    """
    Send DGL batch to device (without line graph).

    Args:
        batch: Tuple (g, lattice, label)
        device: Target device
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Batch on target device
    """
    g, lattice, label = batch

    return (
        g.to(device, non_blocking=non_blocking),
        lattice.to(device, non_blocking=non_blocking),
        label.to(device, non_blocking=non_blocking),
    )


class LMDBDataset(Dataset):
    """
    LMDB-based dataset for ALIGNN.

    Provides efficient data loading for large-scale datasets using LMDB.
    Compatible with reference implementation.
    """

    def __init__(
        self,
        lmdb_path: str,
        line_graph: bool = True,
        ids: Optional[List] = None
    ):
        """
        Initialize LMDB dataset

        Args:
            lmdb_path: Path to LMDB database directory
            line_graph: Whether dataset includes line graphs
            ids: Optional list of IDs to load (for subset)
        """
        super().__init__()

        self.lmdb_path = lmdb_path
        self.line_graph = line_graph
        self.ids = ids if ids is not None else []

        # Open LMDB environment (read-only, no lock for multi-process)
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        # Get dataset length
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]

        # Set batch preparation function
        if line_graph:
            self.prepare_batch = prepare_line_graph_batch
        else:
            self.prepare_batch = prepare_dgl_batch

    def __len__(self):
        """Get dataset length"""
        return self.length

    def __getitem__(self, idx):
        """
        Get item by index

        Returns:
            If line_graph=True: (g, lg, lattice, label)
            If line_graph=False: (g, lattice, label)
        """
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())

        if serialized_data is None:
            raise KeyError(f"Index {idx} not found in LMDB")

        # Deserialize
        data = pk.loads(serialized_data)

        if self.line_graph:
            # Expect: graph, line_graph, lattice, label
            if len(data) == 4:
                return data
            else:
                raise ValueError(f"Expected 4 items for line_graph=True, got {len(data)}")
        else:
            # Expect: graph, lattice, label
            if len(data) == 3:
                return data
            else:
                raise ValueError(f"Expected 3 items for line_graph=False, got {len(data)}")

    def close(self):
        """Close LMDB environment"""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()

    @staticmethod
    def collate(samples: List[Tuple]):
        """
        Collate function for DataLoader (no line graph).

        Args:
            samples: List of (g, lattice, label) tuples

        Returns:
            Batched (graph, lattices, labels)
        """
        graphs, lattices, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)

        # Stack labels
        if len(labels[0].size()) > 0:
            return batched_graph, torch.stack(lattices), torch.stack(labels)
        else:
            return batched_graph, torch.stack(lattices), torch.tensor(labels)

    @staticmethod
    def collate_line_graph(samples: List[Tuple]):
        """
        Collate function for DataLoader (with line graph).

        Args:
            samples: List of (g, lg, lattice, label) tuples

        Returns:
            Batched (graph, line_graph, lattices, labels)
        """
        graphs, line_graphs, lattices, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)

        # Stack labels
        if len(labels[0].size()) > 0:
            return (
                batched_graph,
                batched_line_graph,
                torch.stack(lattices),
                torch.stack(labels),
            )
        else:
            return (
                batched_graph,
                batched_line_graph,
                torch.stack(lattices),
                torch.tensor(labels),
            )


def create_lmdb_dataset(
    graphs: List,
    labels: torch.Tensor,
    lattices: torch.Tensor,
    lmdb_path: str,
    line_graphs: Optional[List] = None,
    map_size: int = 1099511627776  # 1TB default
):
    """
    Create LMDB dataset from graphs and labels.

    Args:
        graphs: List of DGL graphs
        labels: Tensor of labels
        lattices: Tensor of lattice matrices
        lmdb_path: Output LMDB directory path
        line_graphs: Optional list of line graphs
        map_size: Maximum size of LMDB database
    """
    # Create directory
    os.makedirs(lmdb_path, exist_ok=True)

    # Open LMDB environment
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        meminit=False,
        map_async=True,
    )

    print(f"Creating LMDB dataset at {lmdb_path}...")

    with env.begin(write=True) as txn:
        for idx in tqdm(range(len(graphs)), desc="Writing to LMDB"):
            g = graphs[idx]
            label = labels[idx]
            lattice = lattices[idx]

            if line_graphs is not None:
                lg = line_graphs[idx]
                data = (g, lg, lattice, label)
            else:
                data = (g, lattice, label)

            # Serialize and write
            serialized = pk.dumps(data)
            txn.put(f"{idx}".encode(), serialized)

    env.close()
    print(f"LMDB dataset created with {len(graphs)} samples")


def get_lmdb_loaders(
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
    Get train/val/test DataLoaders from LMDB dataset.

    Args:
        lmdb_path: Path to LMDB database
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
    from torch.utils.data import DataLoader, Subset

    # Create dataset
    dataset = LMDBDataset(lmdb_path=lmdb_path, line_graph=line_graph)

    # Split indices
    total_size = len(dataset)
    indices = list(range(total_size))

    # Shuffle
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Calculate sizes
    n_train = int(train_ratio * total_size)
    n_val = int(val_ratio * total_size)
    n_test = int(test_ratio * total_size)

    # Split
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:n_train + n_val + n_test]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Collate function
    collate_fn = dataset.collate_line_graph if line_graph else dataset.collate

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
