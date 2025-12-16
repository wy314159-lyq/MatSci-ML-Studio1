"""
CGCNN - Crystal Graph Convolutional Neural Networks
====================================================

Simplified and improved implementation based on:
Xie & Grossman, Physical Review Letters 120, 145301 (2018)

Features:
- Easy-to-use API
- Automatic data handling
- Configuration presets
- PyQt5 UI integration
- Training monitoring

Quick Start:
-----------
from modules.cgcnn import CIFData, CrystalGraphConvNet, CGCNNTrainer, CGCNNConfig

# Load data
dataset = CIFData('data_directory')

# Create model
config = CGCNNConfig.formation_energy()
structures, _, _ = dataset[0]
model = CrystalGraphConvNet(
    orig_atom_fea_len=structures[0].shape[-1],
    nbr_fea_len=structures[1].shape[-1],
    **config.to_dict()
)

# Train
trainer = CGCNNTrainer(model)
results = trainer.train(train_loader, val_loader, test_loader)
"""

__version__ = "2.0.0"
__author__ = "MatSci-ML Studio (based on Xie & Grossman implementation)"

# Track module availability
CGCNN_AVAILABLE = False
UI_AVAILABLE = False
_IMPORT_ERROR = None

def check_dependencies():
    """
    Check if all required dependencies are available.

    Returns
    -------
    dict
        Dictionary of dependency names and their availability
    """
    deps = {}

    # Core dependencies
    try:
        import torch
        deps['torch'] = torch.__version__
    except (ImportError, OSError, Exception):
        # OSError handles DLL loading failures on Windows
        deps['torch'] = None

    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except (ImportError, OSError, Exception):
        deps['numpy'] = None

    try:
        from pymatgen.core.structure import Structure
        import pymatgen
        try:
            deps['pymatgen'] = pymatgen.__version__
        except AttributeError:
            deps['pymatgen'] = "installed"
    except (ImportError, OSError, Exception):
        deps['pymatgen'] = None

    # Optional dependencies
    try:
        from PyQt5.QtCore import PYQT_VERSION_STR
        deps['PyQt5'] = PYQT_VERSION_STR
    except (ImportError, OSError, Exception):
        deps['PyQt5'] = None

    try:
        import matplotlib
        deps['matplotlib'] = matplotlib.__version__
    except (ImportError, OSError, Exception):
        deps['matplotlib'] = None

    return deps


def get_missing_dependencies():
    """Get list of missing critical dependencies."""
    deps = check_dependencies()
    critical = ['torch', 'numpy', 'pymatgen']
    return [dep for dep in critical if not deps.get(dep)]


# Try to import core modules
try:
    # Core model
    from .model import ConvLayer, CrystalGraphConvNet

    # Data handling
    from .data import (
        GaussianDistance,
        AtomInitializer,
        AtomCustomJSONInitializer,
        CIFData,
        collate_pool,
        get_train_val_test_loader
    )

    # Training
    from .trainer import CGCNNTrainer, Normalizer, AverageMeter

    # Utilities
    from .utils import CGCNNConfig, create_atom_init_json, prepare_sample_data

    # Data preparation
    from .data_matcher import DataMatcher, prepare_cgcnn_data

    # System Information
    from .system_info import SystemInfo, print_system_info

    # Enhanced Visualization
    from .enhanced_visualizer import EnhancedTrainingVisualizer, ProgressInfo

    CGCNN_AVAILABLE = True

except (ImportError, OSError, Exception) as e:
    # Catch ImportError, OSError (DLL loading failures on Windows), and other exceptions
    _IMPORT_ERROR = str(e)
    CGCNN_AVAILABLE = False

    # Create placeholder classes
    class ConvLayer:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class CrystalGraphConvNet:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class GaussianDistance:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class AtomInitializer:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class AtomCustomJSONInitializer:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class CIFData:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    def collate_pool(*args, **kwargs):
        raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    def get_train_val_test_loader(*args, **kwargs):
        raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class CGCNNTrainer:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class Normalizer:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class AverageMeter:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class CGCNNConfig:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    def create_atom_init_json(*args, **kwargs):
        raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    def prepare_sample_data(*args, **kwargs):
        raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class DataMatcher:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    def prepare_cgcnn_data(*args, **kwargs):
        raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class SystemInfo:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    def print_system_info(*args, **kwargs):
        raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class EnhancedTrainingVisualizer:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")

    class ProgressInfo:
        """Placeholder - CGCNN dependencies not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CGCNN requires PyTorch and pymatgen. Error: {_IMPORT_ERROR}")


# UI (optional) - handled separately
try:
    from .ui_widget import CGCNNModule
    UI_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    # Catch ImportError, OSError (DLL loading failures on Windows), and other exceptions
    UI_AVAILABLE = False
    _UI_IMPORT_ERROR = str(e)

    # Create placeholder UI widget
    try:
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PyQt5.QtCore import Qt

        class CGCNNModule(QWidget):
            """Placeholder widget when CGCNN dependencies are not available."""
            def __init__(self, parent=None):
                super().__init__(parent)
                self._init_placeholder_ui()

            def _init_placeholder_ui(self):
                layout = QVBoxLayout(self)
                layout.setAlignment(Qt.AlignCenter)

                # Title
                title = QLabel("CGCNN Module Unavailable")
                title.setStyleSheet("font-size: 18px; font-weight: bold; color: #e74c3c;")
                title.setAlignment(Qt.AlignCenter)
                layout.addWidget(title)

                # Error message
                missing = get_missing_dependencies()
                if missing:
                    error_msg = f"Missing dependencies: {', '.join(missing)}"
                else:
                    error_msg = f"Import error: {_UI_IMPORT_ERROR}"

                error_label = QLabel(error_msg)
                error_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
                error_label.setAlignment(Qt.AlignCenter)
                error_label.setWordWrap(True)
                layout.addWidget(error_label)

                # Instructions
                instructions = QLabel(
                    "\nTo enable CGCNN functionality, please install:\n"
                    "pip install torch pymatgen\n\n"
                    "For GPU support (recommended):\n"
                    "pip install torch --index-url https://download.pytorch.org/whl/cu118"
                )
                instructions.setStyleSheet("font-size: 11px; color: #95a5a6;")
                instructions.setAlignment(Qt.AlignCenter)
                layout.addWidget(instructions)

                layout.addStretch()

    except ImportError:
        # PyQt5 not available either
        class CGCNNModule:
            """Placeholder when PyQt5 is not available."""
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "CGCNNModule requires PyQt5. Install with: pip install PyQt5"
                )


__all__ = [
    # Version
    '__version__',
    '__author__',

    # Availability flags
    'CGCNN_AVAILABLE',
    'UI_AVAILABLE',

    # Model
    'ConvLayer',
    'CrystalGraphConvNet',

    # Data
    'GaussianDistance',
    'AtomInitializer',
    'AtomCustomJSONInitializer',
    'CIFData',
    'collate_pool',
    'get_train_val_test_loader',

    # Training
    'CGCNNTrainer',
    'Normalizer',
    'AverageMeter',

    # Utils
    'CGCNNConfig',
    'create_atom_init_json',
    'prepare_sample_data',

    # Data Preparation
    'DataMatcher',
    'prepare_cgcnn_data',

    # System Info
    'SystemInfo',
    'print_system_info',

    # Visualization
    'EnhancedTrainingVisualizer',
    'ProgressInfo',

    # UI
    'CGCNNModule',

    # Utilities
    'check_dependencies',
    'get_missing_dependencies'
]


def get_version():
    """Get version string."""
    return __version__


def print_info():
    """Print module information and dependency status."""
    print("=" * 60)
    print(f"CGCNN Module v{__version__}")
    print("=" * 60)
    print(f"Author: {__author__}")
    print(f"Based on: Xie & Grossman, PRL 2018")
    print()
    print("Dependencies:")
    print("-" * 60)

    deps = check_dependencies()
    for name, version in deps.items():
        status = f"✓ {version}" if version else "✗ Not installed"
        print(f"  {name:15s} {status}")

    print("=" * 60)

    # Check critical dependencies
    missing = get_missing_dependencies()

    if missing:
        print()
        print("WARNING: Missing critical dependencies:", ', '.join(missing))
        print("Install with: pip install torch numpy pymatgen")
    else:
        print()
        print("All core dependencies satisfied!")

    if not deps.get('PyQt5'):
        print("Note: PyQt5 not available - UI features disabled")

    print("=" * 60)
    print(f"CGCNN Available: {CGCNN_AVAILABLE}")
    print(f"UI Available: {UI_AVAILABLE}")
    print("=" * 60)
