"""
ALIGNN Configuration Module
Configuration classes for ALIGNN training and model setup
"""

from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class ALIGNNConfig(BaseSettings):
    """
    ALIGNN Model Configuration

    Based on reference implementation with sensible defaults
    """

    name: Literal["alignn"] = "alignn"

    # Architecture parameters
    alignn_layers: int = Field(default=4, description="Number of ALIGNN layers")
    gcn_layers: int = Field(default=4, description="Number of GCN layers")
    atom_input_features: int = Field(default=92, description="Atom feature dimension (cgcnn default)")
    edge_input_features: int = Field(default=80, description="Edge feature dimension (RBF bins)")
    triplet_input_features: int = Field(default=40, description="Triplet/angle feature dimension")
    embedding_features: int = Field(default=64, description="Embedding layer dimension")
    hidden_features: int = Field(default=256, description="Hidden layer dimension")
    output_features: int = Field(default=1, description="Output dimension")

    # Activation and normalization
    link: Literal["identity", "log", "logit"] = Field(default="identity", description="Output link function")
    zero_inflated: bool = Field(default=False, description="Use zero-inflated model")
    classification: bool = Field(default=False, description="Classification task")
    num_classes: int = Field(default=2, description="Number of classes for classification")

    # Extra features support
    extra_features: int = Field(default=0, description="Additional lattice/structure features")

    class Config:
        env_prefix = "alignn_model_"


class ALIGNNAtomWiseConfig(BaseSettings):
    """
    ALIGNN AtomWise Model Configuration

    Extended configuration for atomwise predictions (forces, stresses, etc.)
    """

    name: Literal["alignn_atomwise"] = "alignn_atomwise"

    # Architecture parameters (smaller default for atomwise)
    alignn_layers: int = Field(default=2, description="Number of ALIGNN layers")
    gcn_layers: int = Field(default=2, description="Number of GCN layers")
    atom_input_features: int = Field(default=1, description="Atom feature dimension")
    edge_input_features: int = Field(default=80, description="Edge feature dimension")
    triplet_input_features: int = Field(default=40, description="Triplet feature dimension")
    embedding_features: int = Field(default=64, description="Embedding dimension")
    hidden_features: int = Field(default=64, description="Hidden dimension")
    output_features: int = Field(default=1, description="Graph-level output")

    # Gradient and force calculation
    calculate_gradient: bool = Field(default=True, description="Calculate forces via autograd")
    grad_multiplier: int = Field(default=-1, description="Force gradient multiplier")

    # Multi-task learning weights
    graphwise_weight: float = Field(default=1.0, description="Energy/property loss weight")
    gradwise_weight: float = Field(default=1.0, description="Force loss weight")
    stresswise_weight: float = Field(default=0.0, description="Stress loss weight")
    atomwise_weight: float = Field(default=0.0, description="Atomwise property loss weight")
    additional_output_weight: float = Field(default=0.0, description="Additional output loss weight")

    # Output dimensions
    atomwise_output_features: int = Field(default=0, description="Atomwise output dimension")
    additional_output_features: int = Field(default=0, description="Additional output dimension")

    # Physical constraints
    energy_mult_natoms: bool = Field(default=True, description="Multiply energy by number of atoms")
    force_mult_natoms: bool = Field(default=False, description="Multiply forces by number of atoms")
    add_reverse_forces: bool = Field(default=True, description="Add reverse edge forces")

    # Cutoff function
    use_cutoff_function: bool = Field(default=False, description="Use smooth cutoff")
    inner_cutoff: float = Field(default=3.0, description="Cutoff radius (Angstrom)")
    exponent: int = Field(default=5, description="Cutoff function exponent")
    multiply_cutoff: bool = Field(default=False, description="Multiply by cutoff function")

    # Penalty term
    use_penalty: bool = Field(default=True, description="Use distance penalty")
    penalty_factor: float = Field(default=0.1, description="Penalty weight")
    penalty_threshold: float = Field(default=1.0, description="Penalty threshold (Angstrom)")

    # Stress calculation
    stress_multiplier: float = Field(default=1.0, description="Stress multiplier")
    batch_stress: bool = Field(default=True, description="Batch stress calculation")

    # Line graph
    lg_on_fly: bool = Field(default=True, description="Compute line graph on-the-fly")
    include_pos_deriv: bool = Field(default=False, description="Include position derivatives")

    # Classification
    classification: bool = Field(default=False, description="Classification task")
    link: Literal["identity", "log", "logit"] = Field(default="identity", description="Output link")

    # Extra features
    extra_features: int = Field(default=0, description="Extra lattice features")

    class Config:
        env_prefix = "alignn_atomwise_"


class TrainingConfig(BaseSettings):
    """
    ALIGNN Training Configuration

    Complete training configuration based on reference implementation
    """

    # Dataset configuration
    dataset: str = Field(default="user_data", description="Dataset name")
    target: str = Field(default="target", description="Target property name")
    id_tag: str = Field(default="id", description="ID column name")
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = Field(
        default="cgcnn", description="Atom feature type"
    )

    # Graph construction
    neighbor_strategy: Literal["k-nearest", "voronoi", "radius_graph", "radius_graph_jarvis"] = Field(
        default="k-nearest", description="Neighbor finding strategy: k-nearest, voronoi, radius_graph, or radius_graph_jarvis"
    )
    cutoff: float = Field(default=8.0, description="Cutoff radius (Angstrom)")
    cutoff_extra: float = Field(default=3.0, description="Extra cutoff for adjustment (aligned with official reference)")
    max_neighbors: int = Field(default=12, description="Maximum number of neighbors (k-nearest only)")
    use_canonize: bool = Field(default=True, description="Canonize edges (True for robustness and undirected edges)")
    compute_line_graph: bool = Field(default=True, description="Compute line graph for angles")

    # Data splitting
    random_seed: Optional[int] = Field(default=123, description="Random seed")
    train_ratio: Optional[float] = Field(default=0.8, description="Training set ratio")
    val_ratio: Optional[float] = Field(default=0.1, description="Validation set ratio")
    test_ratio: Optional[float] = Field(default=0.1, description="Test set ratio")
    n_train: Optional[int] = Field(default=None, description="Number of training samples")
    n_val: Optional[int] = Field(default=None, description="Number of validation samples")
    n_test: Optional[int] = Field(default=None, description="Number of test samples")
    keep_data_order: bool = Field(default=True, description="Keep original data order (aligned with official reference)")

    # Training parameters
    epochs: int = Field(default=300, description="Number of training epochs")
    batch_size: int = Field(default=64, description="Batch size")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    warmup_steps: int = Field(default=2000, description="Learning rate warmup steps")

    # Optimizer and scheduler
    optimizer: Literal["adamw", "sgd"] = Field(default="adamw", description="Optimizer type")
    scheduler: Literal["onecycle", "step", "none"] = Field(default="onecycle", description="LR scheduler")
    criterion: Literal["mse", "l1", "poisson", "zig"] = Field(default="mse", description="Loss function")

    # Data loading
    num_workers: int = Field(default=4, description="Number of data loading workers")
    pin_memory: bool = Field(default=False, description="Pin memory for data loading")
    use_lmdb: bool = Field(
        default=False,
        description="Use LMDB for efficient data loading. NOTE: Not yet implemented - "
                    "requires lmdb package and StructureDataset from reference_alignn/alignn/data.py"
    )
    save_dataloader: bool = Field(default=False, description="Save dataloader for reuse")

    # Model configuration
    model: ALIGNNConfig = Field(default_factory=lambda: ALIGNNConfig(name="alignn"))

    # Classification
    classification_threshold: Optional[float] = Field(default=None, description="Classification threshold")

    # Preprocessing
    target_multiplication_factor: Optional[float] = Field(
        default=None, description="Multiply target by factor"
    )
    standard_scalar_and_pca: bool = Field(
        default=False,
        description="Apply StandardScaler and PCA to atom features. NOTE: Not yet implemented - "
                    "requires sklearn preprocessing pipeline from reference_alignn/alignn/data.py:140-180"
    )
    normalize_graph_level_loss: bool = Field(
        default=False,
        description="Normalize loss by number of graphs in batch (aligned with official ALIGNN)"
    )

    # Output
    output_dir: str = Field(default=".", description="Output directory")
    filename: str = Field(default="alignn_training", description="Output filename prefix")
    write_checkpoint: bool = Field(default=True, description="Save model checkpoints")
    write_predictions: bool = Field(default=True, description="Save predictions")
    store_outputs: bool = Field(default=True, description="Store outputs")
    log_tensorboard: bool = Field(default=False, description="Log to TensorBoard")

    # Early stopping
    n_early_stopping: Optional[int] = Field(default=None, description="Early stopping patience")

    # Progress
    progress: bool = Field(default=True, description="Show progress bar")

    # Distributed training
    distributed: bool = Field(default=False, description="Use distributed training")
    data_parallel: bool = Field(default=False, description="Use data parallel")

    # Data type
    dtype: str = Field(default="float32", description="Data type for tensors")

    class Config:
        env_prefix = "alignn_train_"

    @classmethod
    def formation_energy(cls):
        """Preset configuration for formation energy prediction"""
        return cls(
            target="formation_energy_peratom",
            epochs=300,
            batch_size=64,
            learning_rate=0.01,
            atom_features="cgcnn"
        )

    @classmethod
    def band_gap(cls):
        """Preset configuration for band gap prediction"""
        return cls(
            target="band_gap",
            epochs=300,
            batch_size=64,
            learning_rate=0.01,
            atom_features="cgcnn"
        )

    @classmethod
    def forces(cls):
        """Preset configuration for force prediction"""
        config = cls(
            target="energy",
            epochs=300,
            batch_size=32,
            learning_rate=0.001
        )
        # Use atomwise model
        config.model = ALIGNNAtomWiseConfig(
            name="alignn_atomwise",
            calculate_gradient=True,
            gradwise_weight=1.0
        )
        return config

    @classmethod
    def quick_test(cls):
        """Quick test configuration with minimal resources"""
        return cls(
            epochs=10,
            batch_size=16,
            learning_rate=0.01,
            n_train=100,
            n_val=20,
            n_test=20,
            num_workers=2
        )


def get_default_config(task: str = "property") -> TrainingConfig:
    """
    Get default configuration for common tasks

    Args:
        task: Task type ("property", "formation_energy", "band_gap", "forces", "quick_test")

    Returns:
        TrainingConfig instance
    """
    task_map = {
        "property": TrainingConfig,
        "formation_energy": TrainingConfig.formation_energy,
        "band_gap": TrainingConfig.band_gap,
        "forces": TrainingConfig.forces,
        "quick_test": TrainingConfig.quick_test
    }

    config_fn = task_map.get(task, TrainingConfig)
    return config_fn()
