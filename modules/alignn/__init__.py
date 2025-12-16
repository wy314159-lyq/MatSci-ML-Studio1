"""
ALIGNN Module
Atomistic Line Graph Neural Network for Material Property Prediction

This module provides a complete implementation of ALIGNN based on the reference
implementation, with an integrated UI for easy use.
"""

__version__ = "2.0.0"
__author__ = "AutoMatFlow Team"

# Track module availability - will be set during import attempt
ALIGNN_AVAILABLE = False
UI_AVAILABLE = False
_IMPORT_ERROR = None
_UI_IMPORT_ERROR = None


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
        # OSError handles DLL load failures on Windows
        deps['torch'] = None

    try:
        import dgl
        deps['dgl'] = dgl.__version__
    except (ImportError, OSError, Exception):
        # OSError handles DLL load failures on Windows
        deps['dgl'] = None

    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except (ImportError, OSError, Exception):
        deps['numpy'] = None

    try:
        from jarvis.core.atoms import Atoms
        deps['jarvis-tools'] = "installed"
    except (ImportError, OSError, Exception):
        deps['jarvis-tools'] = None

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
    critical = ['torch', 'dgl', 'numpy', 'jarvis-tools']
    return [dep for dep in critical if not deps.get(dep)]


def _check_alignn_available():
    """Check if ALIGNN core functionality is available."""
    try:
        import torch
        import dgl
        return True
    except (ImportError, OSError, Exception):
        return False


# Lazy imports for all modules to avoid import failures at startup
def __getattr__(name):
    """Lazy import for all modules to avoid import failures at startup"""
    global _IMPORT_ERROR, ALIGNN_AVAILABLE

    # Config classes (requires pydantic)
    if name == "ALIGNNConfig":
        try:
            from .config import ALIGNNConfig
            return ALIGNNConfig
        except ImportError as e:
            raise ImportError(f"ALIGNNConfig requires pydantic. Error: {e}")

    elif name == "TrainingConfig":
        try:
            from .config import TrainingConfig
            return TrainingConfig
        except ImportError as e:
            raise ImportError(f"TrainingConfig requires pydantic. Error: {e}")

    # Utils (requires torch)
    elif name == "RBFExpansion":
        try:
            from .utils import RBFExpansion
            return RBFExpansion
        except (ImportError, OSError) as e:
            raise ImportError(f"RBFExpansion requires PyTorch. Error: {e}")

    # Graph builder (requires DGL)
    elif name == "ALIGNNGraphBuilder":
        try:
            from .graph_builder import ALIGNNGraphBuilder
            return ALIGNNGraphBuilder
        except (ImportError, OSError) as e:
            raise ImportError(f"ALIGNNGraphBuilder requires DGL. Error: {e}")

    elif name == "ALIGNNDataLoader":
        try:
            from .data_loader import ALIGNNDataLoader
            return ALIGNNDataLoader
        except (ImportError, OSError) as e:
            raise ImportError(f"ALIGNNDataLoader requires DGL. Error: {e}")

    # New data module (CGCNN-style)
    elif name == "CIFData":
        try:
            from .data import CIFData
            return CIFData
        except (ImportError, OSError) as e:
            raise ImportError(f"CIFData requires DGL and jarvis-tools. Error: {e}")

    elif name == "CIFDataPredict":
        try:
            from .data import CIFDataPredict
            return CIFDataPredict
        except (ImportError, OSError) as e:
            raise ImportError(f"CIFDataPredict requires DGL and jarvis-tools. Error: {e}")

    elif name == "JarvisData":
        try:
            from .data import JarvisData
            return JarvisData
        except (ImportError, OSError) as e:
            raise ImportError(f"JarvisData requires DGL and jarvis-tools. Error: {e}")

    elif name == "get_train_val_test_loader":
        try:
            from .data import get_train_val_test_loader
            return get_train_val_test_loader
        except (ImportError, OSError) as e:
            raise ImportError(f"get_train_val_test_loader requires DGL. Error: {e}")

    elif name == "collate_line_graph":
        try:
            from .data import collate_line_graph
            return collate_line_graph
        except (ImportError, OSError) as e:
            raise ImportError(f"collate_line_graph requires DGL. Error: {e}")

    elif name == "load_dataset":
        try:
            from .data import load_dataset
            return load_dataset
        except (ImportError, OSError) as e:
            raise ImportError(f"load_dataset requires DGL. Error: {e}")

    # Data Matcher (no heavy dependencies)
    elif name == "DataMatcher":
        from .data_matcher import DataMatcher
        return DataMatcher

    elif name == "prepare_alignn_data":
        from .data_matcher import prepare_alignn_data
        return prepare_alignn_data

    elif name == "ALIGNN":
        try:
            from .model import ALIGNN
            return ALIGNN
        except (ImportError, OSError) as e:
            raise ImportError(f"ALIGNN model requires PyTorch and DGL. Error: {e}")

    elif name == "ALIGNNAtomWise":
        try:
            from .model import ALIGNNAtomWise
            return ALIGNNAtomWise
        except (ImportError, OSError) as e:
            raise ImportError(f"ALIGNNAtomWise requires PyTorch and DGL. Error: {e}")

    elif name == "ALIGNNTrainer":
        try:
            from .trainer import ALIGNNTrainer
            return ALIGNNTrainer
        except (ImportError, OSError) as e:
            raise ImportError(f"ALIGNNTrainer requires PyTorch and DGL. Error: {e}")

    elif name == "LMDBDataset":
        try:
            from .lmdb_dataset import LMDBDataset
            return LMDBDataset
        except (ImportError, OSError) as e:
            raise ImportError(f"LMDBDataset requires DGL. Error: {e}")

    elif name == "create_lmdb_dataset":
        try:
            from .lmdb_dataset import create_lmdb_dataset
            return create_lmdb_dataset
        except (ImportError, OSError) as e:
            raise ImportError(f"create_lmdb_dataset requires DGL. Error: {e}")

    elif name == "get_lmdb_loaders":
        try:
            from .lmdb_dataset import get_lmdb_loaders
            return get_lmdb_loaders
        except (ImportError, OSError) as e:
            raise ImportError(f"get_lmdb_loaders requires DGL. Error: {e}")

    elif name == "ALIGNN_AVAILABLE":
        return _check_alignn_available()

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# UI (optional) - try to import, create placeholder if fails
# Note: This must handle OSError for DLL loading failures on Windows
try:
    from .ui_widget import ALIGNNModule
    UI_AVAILABLE = True
    ALIGNN_AVAILABLE = _check_alignn_available()
except (ImportError, OSError, Exception) as e:
    # Catch ImportError, OSError (DLL loading failures), and any other exceptions
    UI_AVAILABLE = False
    _UI_IMPORT_ERROR = str(e)
    ALIGNN_AVAILABLE = False

    # Create placeholder UI widget
    try:
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PyQt5.QtCore import Qt

        class ALIGNNModule(QWidget):
            """Placeholder widget when ALIGNN dependencies are not available."""
            def __init__(self, parent=None):
                super().__init__(parent)
                self._init_placeholder_ui()

            def _init_placeholder_ui(self):
                layout = QVBoxLayout(self)
                layout.setAlignment(Qt.AlignCenter)

                # Title
                title = QLabel("ALIGNN Module Unavailable")
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
                    "\nTo enable ALIGNN functionality, please install:\n"
                    "pip install torch dgl jarvis-tools\n\n"
                    "For GPU support (recommended):\n"
                    "pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                    "pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html"
                )
                instructions.setStyleSheet("font-size: 11px; color: #95a5a6;")
                instructions.setAlignment(Qt.AlignCenter)
                layout.addWidget(instructions)

                layout.addStretch()

    except ImportError:
        # PyQt5 not available either
        class ALIGNNModule:
            """Placeholder when PyQt5 is not available."""
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "ALIGNNModule requires PyQt5. Install with: pip install PyQt5"
                )


__all__ = [
    # Version
    '__version__',
    '__author__',

    # Availability flags
    'ALIGNN_AVAILABLE',
    'UI_AVAILABLE',

    # Config (lazy)
    'ALIGNNConfig',
    'TrainingConfig',

    # Graph building (lazy)
    'ALIGNNGraphBuilder',

    # Data loading - new CGCNN-style (lazy)
    'CIFData',
    'CIFDataPredict',
    'JarvisData',
    'get_train_val_test_loader',
    'collate_line_graph',
    'load_dataset',

    # Data Matcher (folder + property table matching)
    'DataMatcher',
    'prepare_alignn_data',

    # Data loading - legacy (lazy)
    'ALIGNNDataLoader',

    # Models (lazy)
    'ALIGNN',
    'ALIGNNAtomWise',

    # Training (lazy)
    'ALIGNNTrainer',

    # LMDB (lazy)
    'LMDBDataset',
    'create_lmdb_dataset',
    'get_lmdb_loaders',

    # Utils (lazy)
    'RBFExpansion',

    # UI
    'ALIGNNModule',

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
    print(f"ALIGNN Module v{__version__}")
    print("=" * 60)
    print(f"Author: {__author__}")
    print()
    print("Dependencies:")
    print("-" * 60)

    deps = check_dependencies()
    for name, version in deps.items():
        status = f"v {version}" if version else "x Not installed"
        print(f"  {name:15s} {status}")

    print("=" * 60)

    # Check critical dependencies
    missing = get_missing_dependencies()

    if missing:
        print()
        print("WARNING: Missing critical dependencies:", ', '.join(missing))
        print("Install with: pip install torch dgl jarvis-tools")
    else:
        print()
        print("All core dependencies satisfied!")

    if not deps.get('PyQt5'):
        print("Note: PyQt5 not available - UI features disabled")

    print("=" * 60)
    print(f"ALIGNN Available: {_check_alignn_available()}")
    print(f"UI Available: {UI_AVAILABLE}")
    print("=" * 60)
