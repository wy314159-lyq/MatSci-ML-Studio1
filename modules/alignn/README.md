# ALIGNN Module v2.0

## Overview

This is a complete reimplementation of the ALIGNN (Atomistic Line Graph Neural Network) module based on the reference implementation from `alignn` package. The module is designed to be easy to use while maintaining full compatibility with the original ALIGNN architecture.

## Features

- **Complete ALIGNN Implementation**: Edge-gated graph convolution with line graph for bond angles
- **Easy-to-Use Interface**: Simplified data loading and training workflow
- **Integrated UI**: PyQt5-based graphical interface matching the application theme
- **Flexible Configuration**: Preset configurations for common tasks
- **Full Training Pipeline**: Data loading, model training, validation, and evaluation
- **Visualization**: Real-time training progress and result visualization

## Architecture

The module consists of the following components:

### 1. Configuration (`config.py`)
- `ALIGNNConfig`: Model architecture configuration
- `TrainingConfig`: Training parameters configuration
- Preset configurations for common tasks:
  - Property prediction
  - Formation energy
  - Band gap
  - Force prediction
  - Quick testing

### 2. Graph Builder (`graph_builder.py`)
- Converts crystal structures to DGL graphs
- Supports k-nearest neighbor graph construction
- Computes bond angles for line graph
- Handles periodic boundary conditions

### 3. Data Loader (`data_loader.py`)
- Loads data from CSV/Excel files
- Builds DGL graphs from structures
- Creates PyTorch DataLoader for training
- Handles train/val/test splitting

### 4. Model (`model.py`)
- **ALIGNN**: Main model class
- **EdgeGatedGraphConv**: Edge-gated graph convolution layer
- **ALIGNNConv**: Combined node and edge update layer
- **RBFExpansion**: Radial basis function for distance features

### 5. Trainer (`trainer.py`)
- Training loop with validation
- Model checkpointing
- Learning rate scheduling
- Progress tracking and callbacks

### 6. UI Widget (`ui_widget.py`)
- Integrated PyQt5 interface
- Data loading interface
- Configuration panel
- Training progress visualization
- Results display

## Installation Requirements

```bash
# Core requirements
pip install torch
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install jarvis-tools
pip install pandas numpy scipy scikit-learn
pip install matplotlib
pip install PyQt5
pip install pydantic pydantic-settings
pip install tqdm
```

## Usage

### Method 1: Using the UI (Recommended)

1. Launch the main application
2. Navigate to the ALIGNN tab
3. Load your data file (CSV/Excel)
4. Select a preset configuration or customize parameters
5. Click "Start Training"

### Method 2: Programmatic Usage

```python
from modules.alignn import (
    TrainingConfig,
    ALIGNNDataLoader,
    ALIGNN,
    ALIGNNTrainer
)

# Create configuration
config = TrainingConfig.formation_energy()

# Load data
data_loader = ALIGNNDataLoader(
    data_path="data.csv",
    target="formation_energy_peratom",
    id_tag="structure_id"
)

# Get data loaders
train_loader, val_loader, test_loader, dataset = data_loader.get_data_loaders(
    batch_size=config.batch_size,
    train_ratio=config.train_ratio,
    val_ratio=config.val_ratio,
    test_ratio=config.test_ratio
)

# Create model
model = ALIGNN(config.model)

# Create trainer
trainer = ALIGNNTrainer(config, model)

# Train
results = trainer.train(train_loader, val_loader, test_loader)
```

## Data Format

Your input data file (CSV or Excel) should contain:

1. **ID column**: Unique identifier for each structure
2. **Target column**: Property values to predict
3. **atoms column**: Crystal structure in JARVIS Atoms dictionary format

Example CSV:
```csv
id,target,atoms
structure_1,2.5,"{""lattice_mat"": [...], ""coords"": [...], ""elements"": [...]}"
structure_2,1.8,"{""lattice_mat"": [...], ""coords"": [...], ""elements"": [...]}"
...
```

## Configuration Presets

### Formation Energy
```python
config = TrainingConfig.formation_energy()
# epochs=300, batch_size=64, lr=0.01
```

### Band Gap
```python
config = TrainingConfig.band_gap()
# epochs=300, batch_size=64, lr=0.01
```

### Forces (with gradient calculation)
```python
config = TrainingConfig.forces()
# Uses ALIGNNAtomWise model with force calculation
```

### Quick Test
```python
config = TrainingConfig.quick_test()
# epochs=10, batch_size=16, n_train=100
```

## Model Architecture

### ALIGNN Layer
```
Input: (g, lg, x, y, z)
  g: Crystal graph
  lg: Line graph (bond angle graph)
  x: Node features (atoms)
  y: Edge features (bonds)
  z: Angle features (triplets)

Operations:
  1. EdgeGatedGraphConv on crystal graph: (x, y) -> (x', y')
  2. EdgeGatedGraphConv on line graph: (y', z) -> (y'', z')

Output: (x', y'', z')
```

### Full Model Pipeline
```
1. Atom Embedding: elements -> node features
2. Edge Embedding: distances -> RBF -> edge features
3. Angle Embedding: bond cosines -> RBF -> angle features
4. ALIGNN Layers: Update x, y, z (default: 4 layers)
5. GCN Layers: Update x, y (default: 4 layers)
6. Global Pooling: Aggregate node features
7. Output Layer: Predict property
```

## Key Parameters

### Model Parameters
- `alignn_layers`: Number of ALIGNN layers (default: 4)
- `gcn_layers`: Number of GCN layers (default: 4)
- `hidden_features`: Hidden dimension (default: 256)
- `edge_input_features`: RBF bins for distances (default: 80)
- `triplet_input_features`: RBF bins for angles (default: 40)

### Graph Construction
- `cutoff`: Cutoff radius in Angstrom (default: 8.0)
- `max_neighbors`: Max neighbors per atom (default: 12)
- `neighbor_strategy`: k-nearest, voronoi, radius_graph (default: k-nearest)
- `use_canonize`: Canonize edges (default: True)

### Training Parameters
- `epochs`: Training epochs (default: 300)
- `batch_size`: Batch size (default: 64)
- `learning_rate`: Learning rate (default: 0.01)
- `optimizer`: adamw, sgd (default: adamw)
- `scheduler`: onecycle, step, none (default: onecycle)

## Output Files

Training produces the following files in `output_dir`:

- `best_model.pt`: Best model checkpoint
- `current_model.pt`: Latest model checkpoint
- `final_model.pt`: Final model after training
- `training_history.json`: Training curves
- `config.json`: Complete configuration

## Comparison with Original ALIGNN

### Maintained Features
- ✓ Complete ALIGNN architecture
- ✓ Edge-gated graph convolution
- ✓ Line graph for bond angles
- ✓ RBF expansion for distances
- ✓ Flexible atom features (cgcnn default)
- ✓ OneCycleLR scheduler
- ✓ Model checkpointing

### Simplifications
- Removed LMDB dataset support (for simplicity)
- Removed distributed training (can be added if needed)
- Removed atomwise predictions (focused on graph-level)
- Simplified to property prediction only

### Enhancements
- ✓ Integrated PyQt5 UI
- ✓ Simplified data loading
- ✓ Better progress tracking
- ✓ Real-time visualization
- ✓ Preset configurations

## Troubleshooting

### DGL Import Error
```bash
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### JARVIS Import Error
```bash
pip install jarvis-tools
```

### CUDA Out of Memory
- Reduce batch size
- Reduce model size (hidden_features, alignn_layers, gcn_layers)
- Use CPU instead: Set device to "cpu"

### Data Loading Error
- Ensure "atoms" column contains valid JARVIS Atoms dictionaries
- Check that target column has numeric values
- Verify ID column contains unique identifiers

## Performance Tips

1. **Use GPU**: 10-100x faster than CPU
2. **Batch Size**: Larger batch size = faster training (if memory allows)
3. **Number of Workers**: Set num_workers=4 for data loading
4. **Model Size**: Start small (2 ALIGNN + 2 GCN layers) for testing
5. **Quick Test**: Use quick_test preset for initial validation

## Citation

If you use this module, please cite the original ALIGNN paper:

```
@article{choudhary2021atomistic,
  title={Atomistic Line Graph Neural Network for improved materials property predictions},
  author={Choudhary, Kamal and DeCost, Brian},
  journal={npj Computational Materials},
  volume={7},
  number={1},
  pages={185},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## License

This module is part of AutoMatFlow and follows the same license as the main project.

## Contact

For issues and questions, please open an issue in the main project repository.

---

**Version**: 2.0.0
**Last Updated**: 2025
**Status**: Production Ready
