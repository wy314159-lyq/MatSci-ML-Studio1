"""
CGCNN Utilities
Configuration presets and helper functions
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CGCNNConfig:
    """Configuration for CGCNN model and training."""

    # Model architecture
    atom_fea_len: int = 64
    nbr_fea_len: int = 41  # From Gaussian expansion
    n_conv: int = 3
    h_fea_len: int = 128
    n_h: int = 1
    classification: bool = False

    # Data processing
    max_num_nbr: int = 12
    radius: float = 8.0
    dmin: float = 0.0
    step: float = 0.2

    # Training
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.9
    optimizer: str = 'SGD'
    lr_milestones: list = None

    # Data split
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    def __post_init__(self):
        if self.lr_milestones is None:
            self.lr_milestones = [100]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'atom_fea_len': self.atom_fea_len,
            'nbr_fea_len': self.nbr_fea_len,
            'n_conv': self.n_conv,
            'h_fea_len': self.h_fea_len,
            'n_h': self.n_h,
            'classification': self.classification,
            'max_num_nbr': self.max_num_nbr,
            'radius': self.radius,
            'dmin': self.dmin,
            'step': self.step,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'optimizer': self.optimizer,
            'lr_milestones': self.lr_milestones,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def formation_energy(cls):
        """
        Preset for formation energy prediction.

        Optimized hyperparameters based on the original paper.
        """
        return cls(
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
            n_h=1,
            epochs=30,
            learning_rate=0.01,
            batch_size=256,
            classification=False
        )

    @classmethod
    def band_gap(cls):
        """
        Preset for band gap prediction.
        """
        return cls(
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
            n_h=1,
            epochs=30,
            learning_rate=0.01,
            batch_size=256,
            classification=False
        )

    @classmethod
    def classification_task(cls):
        """
        Preset for classification tasks (e.g., metal/semiconductor).
        """
        return cls(
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
            n_h=1,
            epochs=30,
            learning_rate=0.01,
            batch_size=256,
            classification=True
        )

    @classmethod
    def quick_test(cls):
        """
        Quick test configuration for debugging.
        """
        return cls(
            atom_fea_len=32,
            n_conv=2,
            h_fea_len=64,
            n_h=1,
            epochs=5,
            learning_rate=0.01,
            batch_size=32,
            max_num_nbr=8,
            radius=6.0
        )

    @classmethod
    def gpu_optimized(cls):
        """
        GPU-optimized configuration for maximum GPU utilization.

        Key optimizations:
        - Larger batch size (512) to reduce data loading overhead
        - This keeps GPU busy by reducing the ratio of data loading time to compute time
        - Requires GPU with 8GB+ VRAM
        - Best for large datasets (10000+ structures)

        Performance comparison (example):
        - batch_size=256: Data loading 89%, GPU compute 11%
        - batch_size=512: Data loading 70%, GPU compute 30% (estimated)
        - batch_size=1024: Data loading 50%, GPU compute 50% (estimated)
        """
        return cls(
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
            n_h=1,
            epochs=30,
            learning_rate=0.01,
            batch_size=512,  # Larger batch for better GPU utilization
            classification=False
        )


def get_official_atom_init_path():
    """
    Get path to official atom_init.json in CGCNN module directory.

    Returns the path to the official 92-dimensional atom feature vectors
    for elements 1-100, based on the original CGCNN implementation.

    Returns
    -------
    str or None
        Path to official atom_init.json, or None if not found
    """
    import os
    module_dir = os.path.dirname(os.path.abspath(__file__))
    official_path = os.path.join(module_dir, 'atom_init.json')
    if os.path.exists(official_path):
        return official_path
    return None


def create_atom_init_json(output_path='atom_init.json'):
    """
    Copy official atom_init.json to output path.

    This function copies the official CGCNN atom initialization file which
    contains 92-dimensional feature vectors for elements 1-100.

    NOTE: Previous versions created simple one-hot encoding, but this was
    NOT consistent with the original CGCNN paper. We now use the official
    atom features for better prediction accuracy.

    Parameters
    ----------
    output_path : str
        Path to save the JSON file
    """
    import shutil

    official_path = get_official_atom_init_path()

    if official_path is not None:
        shutil.copy(official_path, output_path)
        print(f"Copied official atom_init.json to: {output_path}")
        print(f"Contains 100 elements with 92-dim features (official CGCNN)")
    else:
        # Fallback: create simple one-hot encoding (not recommended)
        import json
        print("WARNING: Official atom_init.json not found!")
        print("Creating simple one-hot encoding (NOT recommended for production)")

        atom_init = {}
        elements = list(range(1, 101))  # H to Fm (elements 1-100)

        for elem in elements:
            features = [0.0] * 92
            if elem <= 92:
                features[elem - 1] = 1.0
            atom_init[str(elem)] = features

        with open(output_path, 'w') as f:
            json.dump(atom_init, f, indent=2)

        print(f"Created fallback atom_init.json: {output_path}")
        print(f"Contains {len(atom_init)} elements with 92-dim one-hot features")
        print("For best results, use official atom_init.json from CGCNN module")


def prepare_sample_data(cif_dir, property_file, output_dir):
    """
    Prepare data directory in CGCNN format.

    Uses the official atom_init.json from the CGCNN module directory,
    which contains 92-dimensional feature vectors for elements 1-100.

    Parameters
    ----------
    cif_dir : str
        Directory containing CIF files
    property_file : str
        CSV file with structure IDs and properties
    output_dir : str
        Output directory for organized data
    """
    import os
    import shutil
    import pandas as pd
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load property file
    df = pd.read_csv(property_file)

    # Assuming first column is ID, second is property
    if df.shape[1] < 2:
        raise ValueError("Property file must have at least 2 columns (ID, property)")

    # Create id_prop.csv
    id_prop_path = output_path / 'id_prop.csv'
    df.iloc[:, :2].to_csv(id_prop_path, index=False, header=False)

    # Copy CIF files
    cif_path = Path(cif_dir)
    copied_count = 0
    for idx, row in df.iterrows():
        cif_id = str(row.iloc[0])
        cif_file = cif_path / f"{cif_id}.cif"

        if cif_file.exists():
            shutil.copy(cif_file, output_path / f"{cif_id}.cif")
            copied_count += 1
        else:
            print(f"Warning: CIF file not found for {cif_id}")

    # Copy official atom_init.json (prioritize official over creating new)
    atom_init_path = output_path / 'atom_init.json'
    official_atom_init = get_official_atom_init_path()

    if official_atom_init is not None:
        shutil.copy(official_atom_init, atom_init_path)
        print(f"Using official atom_init.json (92-dim features for elements 1-100)")
    else:
        # Fallback to creating one (not recommended)
        create_atom_init_json(str(atom_init_path))

    print(f"\nData prepared in: {output_dir}")
    print(f"  - {len(df)} structures in property file")
    print(f"  - {copied_count} CIF files copied")
    print(f"  - id_prop.csv created")
    print(f"  - atom_init.json configured")
