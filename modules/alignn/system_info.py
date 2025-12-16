"""
System Information Checker for ALIGNN
Checks hardware and software environment requirements for ALIGNN training
"""

import platform
import sys
import os
import subprocess
from typing import Dict, List, Tuple


class ALIGNNSystemInfo:
    """
    System information checker for ALIGNN training requirements.

    Checks:
    - Python version
    - PyTorch installation and CUDA availability
    - System resources (CPU, RAM, GPU)
    - Dependencies
    - Disk space
    - ALIGNN specific requirements and limitations
    """

    def __init__(self):
        self.info = {}
        self.warnings = []
        self.errors = []

    def check_all(self) -> Dict:
        """
        Run all system checks.

        Returns
        -------
        dict
            Complete system information and status
        """
        self.check_python()
        self.check_platform()
        self.check_pytorch()
        self.check_cuda()
        self.check_dependencies()
        self.check_system_resources()
        self.check_disk_space()
        self.check_alignn_specific()

        return {
            'info': self.info,
            'warnings': self.warnings,
            'errors': self.errors,
            'status': 'ok' if len(self.errors) == 0 else 'error'
        }

    def check_python(self):
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        self.info['python_version'] = version_str
        self.info['python_executable'] = sys.executable

        # Recommend Python 3.7+
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            self.warnings.append(
                f"Python {version_str} detected. Python 3.7+ is recommended for ALIGNN."
            )

    def check_platform(self):
        """Check operating system information."""
        self.info['os'] = platform.system()
        self.info['os_version'] = platform.version()
        self.info['architecture'] = platform.machine()
        self.info['processor'] = platform.processor()

    def check_pytorch(self):
        """Check PyTorch installation."""
        try:
            import torch
            self.info['pytorch_version'] = torch.__version__
            self.info['pytorch_installed'] = True

            # Check PyTorch build info
            self.info['pytorch_cuda_compiled'] = torch.cuda.is_available()

        except ImportError:
            self.info['pytorch_installed'] = False
            self.errors.append(
                "PyTorch not installed! Install with: pip install torch"
            )

    def check_cuda(self):
        """Check CUDA availability and GPU information."""
        try:
            import torch

            if torch.cuda.is_available():
                self.info['cuda_available'] = True
                self.info['cuda_version_pytorch'] = torch.version.cuda
                self.info['cudnn_version'] = torch.backends.cudnn.version()
                self.info['gpu_count'] = torch.cuda.device_count()

                # Check actual CUDA installation version
                try:
                    result = subprocess.run(['nvcc', '--version'],
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        output = result.stdout
                        # Extract version from nvcc output
                        for line in output.split('\n'):
                            if 'Cuda compilation tools, release' in line:
                                # Extract version number (e.g., "release 12.8, V12.8.93")
                                import re
                                match = re.search(r'release\s+(\d+\.\d+)', line)
                                if match:
                                    self.info['cuda_version_system'] = match.group(1)
                                    break
                        else:
                            self.info['cuda_version_system'] = 'Unknown (nvcc output parsing failed)'
                    else:
                        self.info['cuda_version_system'] = 'Unknown (nvcc failed)'
                except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                    self.info['cuda_version_system'] = 'Unknown (nvcc not found or failed)'

                # Get GPU details
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)

                    gpu_info.append({
                        'id': i,
                        'name': gpu_name,
                        'memory_gb': f"{gpu_memory_gb:.2f}",
                        'memory_bytes': gpu_memory
                    })

                self.info['gpus'] = gpu_info

                # Check GPU memory for ALIGNN
                min_memory_gb = 6  # ALIGNN needs more memory than CGCNN
                for gpu in gpu_info:
                    if float(gpu['memory_gb']) < min_memory_gb:
                        self.warnings.append(
                            f"GPU {gpu['id']} ({gpu['name']}) has only "
                            f"{gpu['memory_gb']}GB memory. "
                            f"At least {min_memory_gb}GB recommended for ALIGNN training."
                        )
            else:
                self.info['cuda_available'] = False
                self.info['gpu_count'] = 0
                self.warnings.append(
                    "CUDA not available. ALIGNN training will use CPU (much slower)."
                )

        except ImportError:
            self.info['cuda_available'] = False
            self.info['gpu_count'] = 0

    def check_dependencies(self):
        """Check required dependencies for ALIGNN."""
        dependencies = {
            # Core dependencies
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'scipy': 'SciPy',
            'pandas': 'Pandas',

            # Materials science - required for ALIGNN
            'pymatgen': 'Pymatgen',
            'jarvis': 'JARVIS-Tools (Required for ALIGNN)',
            'ase': 'ASE (Atomic Simulation Environment)',

            # Deep learning - critical for ALIGNN
            'dgl': 'DGL (Deep Graph Library - Critical for ALIGNN)',
            'sklearn': 'Scikit-learn',

            # UI
            'PyQt5': 'PyQt5 (for UI)',

            # Visualization
            'matplotlib': 'Matplotlib',
            'seaborn': 'Seaborn',

            # Data processing
            'tqdm': 'tqdm (progress bars)',
            'pydantic': 'Pydantic (data validation)',

            # Others
            'psutil': 'psutil (system monitoring)',
            'openpyxl': 'openpyxl (Excel support)',
        }

        installed = {}
        missing = []

        for module_name, display_name in dependencies.items():
            try:
                if module_name == 'PyQt5':
                    from PyQt5.QtCore import PYQT_VERSION_STR
                    installed[display_name] = PYQT_VERSION_STR
                elif module_name == 'sklearn':
                    import sklearn
                    version = getattr(sklearn, '__version__', 'installed')
                    installed[display_name] = version
                elif module_name == 'jarvis':
                    import jarvis
                    version = getattr(jarvis, '__version__', 'installed')
                    installed[display_name] = version
                elif module_name == 'dgl':
                    # DGL can have import issues, be more careful
                    try:
                        import dgl
                        version = getattr(dgl, '__version__', 'installed')
                        installed[display_name] = version
                    except Exception as e:
                        missing.append(f"{display_name} (Error: {str(e)[:50]}...)")
                        continue
                else:
                    mod = __import__(module_name)
                    version = getattr(mod, '__version__', 'installed')
                    installed[display_name] = version
            except ImportError as e:
                missing.append(display_name)
            except Exception as e:
                missing.append(f"{display_name} (Error: {str(e)[:50]}...)")

        self.info['dependencies_installed'] = installed
        self.info['dependencies_missing'] = missing

        # Check critical dependencies for ALIGNN
        critical = ['PyTorch', 'NumPy', 'JARVIS-Tools (Required for ALIGNN)',
                   'DGL (Deep Graph Library - Critical for ALIGNN)', 'Pymatgen']
        for dep in critical:
            if dep in missing:
                self.errors.append(f"Critical ALIGNN dependency missing: {dep}")

        # Add environment paths
        self.info['python_path'] = sys.path
        self.info['environment_variables'] = {
            'PATH': os.environ.get('PATH', 'N/A'),
            'CUDA_HOME': os.environ.get('CUDA_HOME', 'Not set'),
            'CUDA_PATH': os.environ.get('CUDA_PATH', 'Not set'),
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', 'Not set (Linux only)'),
        }

    def check_system_resources(self):
        """Check system CPU and RAM."""
        try:
            import psutil
        except ImportError:
            self.warnings.append("psutil not installed - cannot check system resources")
            return

        # CPU info
        self.info['cpu_count'] = psutil.cpu_count(logical=False)
        self.info['cpu_count_logical'] = psutil.cpu_count(logical=True)

        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self.info['cpu_freq_mhz'] = f"{cpu_freq.current:.0f}"
        except:
            pass

        # RAM info
        memory = psutil.virtual_memory()
        self.info['ram_total_gb'] = f"{memory.total / (1024**3):.2f}"
        self.info['ram_available_gb'] = f"{memory.available / (1024**3):.2f}"
        self.info['ram_percent_used'] = memory.percent

        # Check RAM requirements for ALIGNN
        min_ram_gb = 16  # ALIGNN needs more RAM due to graph processing
        total_ram_gb = memory.total / (1024**3)
        if total_ram_gb < min_ram_gb:
            self.warnings.append(
                f"System has {total_ram_gb:.1f}GB RAM. "
                f"At least {min_ram_gb}GB recommended for ALIGNN training with graphs."
            )

        # Check available RAM
        if memory.percent > 80:
            self.warnings.append(
                f"RAM usage is high ({memory.percent}%). "
                f"Close other applications before ALIGNN training."
            )

    def check_disk_space(self):
        """Check available disk space."""
        try:
            import psutil

            # Get disk usage for current directory
            if platform.system() == 'Windows':
                disk_path = 'C:\\'
            else:
                disk_path = '/'

            disk = psutil.disk_usage(disk_path)
            self.info['disk_total_gb'] = f"{disk.total / (1024**3):.2f}"
            self.info['disk_free_gb'] = f"{disk.free / (1024**3):.2f}"
            self.info['disk_percent_used'] = disk.percent

            # Check disk space for ALIGNN
            min_free_gb = 20  # ALIGNN datasets and models can be large
            free_gb = disk.free / (1024**3)
            if free_gb < min_free_gb:
                self.warnings.append(
                    f"Low disk space: {free_gb:.1f}GB free. "
                    f"At least {min_free_gb}GB recommended for ALIGNN datasets and models."
                )
        except Exception as e:
            self.info['disk_total_gb'] = "N/A"
            self.info['disk_free_gb'] = "N/A"
            self.info['disk_percent_used'] = "N/A"

    def check_alignn_specific(self):
        """Check ALIGNN-specific requirements and limitations."""
        alignn_info = {}

        # Element support limitation
        alignn_info['element_support'] = {
            'max_atomic_number': 92,
            'supported_elements': 'Z=1 (H) to Z=92 (U)',
            'limitation_reason': '92-dimensional CGCNN features, matching reference implementation',
            'superheavy_elements': 'Z>92 will cause IndexError - by design, not a bug',
            'coverage': '99.9% of real materials use Z≤92'
        }

        # Graph construction strategies
        alignn_info['graph_construction'] = {
            'strategies': ['k-nearest (simplified)', 'radius (reference)'],
            'default': 'k-nearest',
            'radius_method': 'Uses reciprocal space supercell search for exact periodic boundaries',
            'k_nearest_method': 'Simplified approach, faster but less precise for edge cases'
        }

        # Model architecture
        alignn_info['architecture'] = {
            'node_features': '92-dimensional atom features',
            'edge_features': '80-dimensional RBF expansion',
            'angle_features': '40-dimensional angle features',
            'normalization': 'BatchNorm1d (matches reference)',
            'activation': 'SiLU',
            'readout': 'AvgPooling'
        }

        # Memory requirements (rough estimates)
        alignn_info['memory_estimates'] = {
            'small_dataset': '< 1000 structures: 4-8GB GPU memory',
            'medium_dataset': '1000-10000 structures: 8-16GB GPU memory',
            'large_dataset': '> 10000 structures: 16GB+ GPU memory',
            'note': 'Actual usage depends on structure size and batch size'
        }

        # Training recommendations
        alignn_info['training_recommendations'] = {
            'batch_size': 'Start with 32-64, adjust based on GPU memory',
            'learning_rate': '0.001-0.01 with OneCycleLR scheduler',
            'epochs': '200-500 depending on dataset size',
            'early_stopping': 'Recommended with patience=50',
            'validation_split': '10-20% for validation'
        }

        self.info['alignn_specific'] = alignn_info

        # Add specific warnings for ALIGNN
        try:
            # Check if DGL is properly installed
            import dgl
            dgl_version = getattr(dgl, '__version__', 'unknown')
            self.info['dgl_version'] = dgl_version
        except Exception as e:
            self.errors.append(f"DGL import failed: {e}. ALIGNN requires DGL for graph processing.")

        try:
            # Check if jarvis is available for atom features
            from jarvis.core.specie import chem_data
            jarvis_elements = len(chem_data)
            self.info['jarvis_elements_available'] = jarvis_elements
        except ImportError:
            self.warnings.append(
                "JARVIS-Tools not available. Will use fallback 92-dimensional one-hot encoding."
            )
        except Exception as e:
            self.warnings.append(f"JARVIS element data issue: {e}")

    def get_summary(self) -> str:
        """Get formatted summary of system information for ALIGNN."""
        lines = []
        lines.append("=" * 70)
        lines.append("ALIGNN System Information & Environment Check")
        lines.append("=" * 70)

        # Platform
        lines.append("\n[Platform]")
        lines.append(f"  OS: {self.info.get('os', 'Unknown')} {self.info.get('os_version', '')}")
        lines.append(f"  Architecture: {self.info.get('architecture', 'Unknown')}")
        lines.append(f"  Processor: {self.info.get('processor', 'Unknown')}")

        # Python
        lines.append("\n[Python]")
        lines.append(f"  Version: {self.info.get('python_version', 'Unknown')}")
        lines.append(f"  Executable: {self.info.get('python_executable', 'Unknown')}")

        # PyTorch
        lines.append("\n[PyTorch]")
        if self.info.get('pytorch_installed'):
            lines.append(f"  Version: {self.info.get('pytorch_version', 'Unknown')}")
            lines.append(f"  CUDA Compiled: {self.info.get('pytorch_cuda_compiled', False)}")
            lines.append(f"  CUDA Available: {self.info.get('cuda_available', False)}")
            if self.info.get('cuda_available'):
                pytorch_cuda = self.info.get('cuda_version_pytorch', 'Unknown')
                system_cuda = self.info.get('cuda_version_system', 'Not detected')
                lines.append(f"  CUDA Version (PyTorch): {pytorch_cuda}")
                lines.append(f"  CUDA Version (System): {system_cuda}")
                lines.append(f"  cuDNN Version: {self.info.get('cudnn_version', 'Unknown')}")
        else:
            lines.append("  NOT INSTALLED")

        # GPU
        lines.append("\n[GPU - ALIGNN Recommendations]")
        gpu_count = self.info.get('gpu_count', 0)
        lines.append(f"  Count: {gpu_count}")
        if gpu_count > 0:
            for gpu in self.info.get('gpus', []):
                lines.append(f"  GPU {gpu['id']}: {gpu['name']}")
                lines.append(f"    Memory: {gpu['memory_gb']} GB")
                memory_gb = float(gpu['memory_gb'])
                if memory_gb >= 16:
                    lines.append(f"    Status: Excellent for large ALIGNN datasets")
                elif memory_gb >= 8:
                    lines.append(f"    Status: Good for medium ALIGNN datasets")
                elif memory_gb >= 4:
                    lines.append(f"    Status: Suitable for small ALIGNN datasets")
                else:
                    lines.append(f"    Status: May be insufficient for ALIGNN training")
        else:
            lines.append("  No GPU detected - ALIGNN training will be VERY slow on CPU")

        # System Resources
        lines.append("\n[System Resources - ALIGNN Requirements]")
        lines.append(f"  CPU Cores (Physical): {self.info.get('cpu_count', 'Unknown')}")
        lines.append(f"  CPU Cores (Logical): {self.info.get('cpu_count_logical', 'Unknown')}")
        if 'cpu_freq_mhz' in self.info:
            lines.append(f"  CPU Frequency: {self.info['cpu_freq_mhz']} MHz")

        ram_total = self.info.get('ram_total_gb', 'Unknown')
        lines.append(f"  RAM Total: {ram_total} GB")
        lines.append(f"  RAM Available: {self.info.get('ram_available_gb', 'Unknown')} GB")
        lines.append(f"  RAM Usage: {self.info.get('ram_percent_used', 'Unknown')}%")

        # RAM recommendation for ALIGNN
        try:
            ram_gb = float(ram_total)
            if ram_gb >= 32:
                lines.append(f"  RAM Status: Excellent for large ALIGNN graphs")
            elif ram_gb >= 16:
                lines.append(f"  RAM Status: Good for most ALIGNN tasks")
            elif ram_gb >= 8:
                lines.append(f"  RAM Status: Minimum for small ALIGNN datasets")
            else:
                lines.append(f"  RAM Status: Insufficient for ALIGNN training")
        except (ValueError, TypeError):
            pass

        # Disk Space
        lines.append("\n[Disk Space]")
        lines.append(f"  Total: {self.info.get('disk_total_gb', 'Unknown')} GB")
        lines.append(f"  Free: {self.info.get('disk_free_gb', 'Unknown')} GB")
        lines.append(f"  Usage: {self.info.get('disk_percent_used', 'Unknown')}%")

        # ALIGNN Specific Information
        alignn_info = self.info.get('alignn_specific', {})

        lines.append("\n[ALIGNN Element Support]")
        element_info = alignn_info.get('element_support', {})
        lines.append(f"  Supported Elements: {element_info.get('supported_elements', 'Z=1 to Z=92')}")
        lines.append(f"  Max Atomic Number: {element_info.get('max_atomic_number', 92)}")
        lines.append(f"  Feature Dimension: 92D CGCNN features")
        lines.append(f"  Coverage: {element_info.get('coverage', '99.9% of real materials')}")
        lines.append(f"  IMPORTANT: Elements with Z>92 will cause IndexError")
        lines.append(f"             This is by design, not a bug - matches reference ALIGNN")

        lines.append("\n[ALIGNN Architecture]")
        arch_info = alignn_info.get('architecture', {})
        lines.append(f"  Node Features: {arch_info.get('node_features', '92-dimensional')}")
        lines.append(f"  Edge Features: {arch_info.get('edge_features', '80-dimensional RBF')}")
        lines.append(f"  Angle Features: {arch_info.get('angle_features', '40-dimensional')}")
        lines.append(f"  Normalization: {arch_info.get('normalization', 'BatchNorm1d')}")
        lines.append(f"  Activation: {arch_info.get('activation', 'SiLU')}")

        lines.append("\n[ALIGNN Graph Construction]")
        graph_info = alignn_info.get('graph_construction', {})
        strategies = graph_info.get('strategies', ['k-nearest', 'radius'])
        lines.append(f"  Available Strategies: {', '.join(strategies)}")
        lines.append(f"  Default Strategy: {graph_info.get('default', 'k-nearest')}")
        lines.append(f"  Note: radius method provides exact periodic boundaries")

        # Critical Dependencies
        lines.append("\n[Critical ALIGNN Dependencies]")
        installed = self.info.get('dependencies_installed', {})
        critical_deps = ['PyTorch', 'DGL (Deep Graph Library - Critical for ALIGNN)',
                        'JARVIS-Tools (Required for ALIGNN)', 'NumPy', 'Pymatgen']
        for dep in critical_deps:
            if dep in installed:
                lines.append(f"  [OK] {dep}: {installed[dep]}")
            else:
                lines.append(f"  [MISSING] {dep}")

        # Other Dependencies
        lines.append("\n[Other Dependencies]")
        other_deps = ['SciPy', 'Pandas', 'ASE (Atomic Simulation Environment)',
                     'Scikit-learn', 'PyQt5 (for UI)', 'Matplotlib']
        for dep in other_deps:
            if dep in installed:
                lines.append(f"  [OK] {dep}: {installed[dep]}")
            else:
                lines.append(f"  [MISSING] {dep}")

        # Memory Estimates
        lines.append("\n[ALIGNN Memory Estimates]")
        memory_info = alignn_info.get('memory_estimates', {})
        for key, value in memory_info.items():
            if key != 'note':
                lines.append(f"  {key}: {value}")
        if 'note' in memory_info:
            lines.append(f"  Note: {memory_info['note']}")

        # Training Recommendations
        lines.append("\n[ALIGNN Training Recommendations]")
        training_info = alignn_info.get('training_recommendations', {})
        for key, value in training_info.items():
            lines.append(f"  {key}: {value}")

        # DGL Specific Info
        if 'dgl_version' in self.info:
            lines.append(f"\n[DGL Information]")
            lines.append(f"  Version: {self.info['dgl_version']}")
            lines.append(f"  Status: Successfully imported")

        # JARVIS Specific Info
        if 'jarvis_elements_available' in self.info:
            lines.append(f"\n[JARVIS Information]")
            lines.append(f"  Elements Available: {self.info['jarvis_elements_available']}")
            lines.append(f"  Status: Will use JARVIS features")
        else:
            lines.append(f"\n[JARVIS Information]")
            lines.append(f"  Status: Not available - using fallback features")

        # Warnings
        if self.warnings:
            lines.append("\n[Warnings]")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")

        # Errors
        if self.errors:
            lines.append("\n[Errors]")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")

        # Status
        lines.append("\n" + "=" * 70)
        if self.errors:
            lines.append("STATUS: ERRORS FOUND")
            lines.append("Action: Please fix errors before ALIGNN training")
        elif self.warnings:
            lines.append("STATUS: WARNINGS PRESENT")
            lines.append("Action: ALIGNN may work but performance may be limited")
        else:
            lines.append("STATUS: ALL CHECKS PASSED")
            lines.append("Action: System ready for ALIGNN training")

        lines.append("\nIMPORTANT ALIGNN LIMITATIONS:")
        lines.append("- Only supports elements Z=1 to Z=92 (H to U)")
        lines.append("- Materials with superheavy elements (Z>92) will fail")
        lines.append("- This matches reference ALIGNN behavior")
        lines.append("- 99.9% of real materials are supported")
        lines.append("=" * 70)

        return "\n".join(lines)

    def get_recommendations(self) -> List[str]:
        """Get ALIGNN-specific system improvement recommendations."""
        recommendations = []

        # GPU recommendation
        if not self.info.get('cuda_available'):
            recommendations.append(
                "CRITICAL: Use system with NVIDIA GPU for ALIGNN training. "
                "CPU training is extremely slow for graph neural networks."
            )

        # GPU memory recommendation
        gpu_info = self.info.get('gpus', [])
        if gpu_info:
            for gpu in gpu_info:
                memory_gb = float(gpu['memory_gb'])
                if memory_gb < 6:
                    recommendations.append(
                        f"GPU {gpu['id']} has only {gpu['memory_gb']}GB memory. "
                        f"Consider upgrading to 8GB+ for comfortable ALIGNN training."
                    )

        # RAM recommendation
        try:
            ram_gb = float(self.info.get('ram_total_gb', 0))
            if ram_gb < 16:
                recommendations.append(
                    f"System has {ram_gb:.1f}GB RAM. "
                    f"Consider 16GB+ for ALIGNN graph processing and large datasets."
                )
        except (ValueError, TypeError):
            pass

        # Disk space recommendation
        try:
            free_gb = float(self.info.get('disk_free_gb', 0))
            if free_gb < 20:
                recommendations.append(
                    f"Free disk space: {free_gb:.1f}GB. "
                    f"Consider having 20GB+ free for ALIGNN datasets and models."
                )
        except (ValueError, TypeError):
            pass

        # Dependencies
        missing = self.info.get('dependencies_missing', [])
        critical_missing = []
        for dep in missing:
            if any(key in dep for key in ['DGL', 'JARVIS', 'PyTorch']):
                critical_missing.append(dep)

        if critical_missing:
            recommendations.append(
                f"CRITICAL: Install missing ALIGNN dependencies first: {', '.join(critical_missing)}"
            )

        # ALIGNN specific recommendations
        recommendations.append(
            "For optimal ALIGNN performance: use datasets with structures containing only Z≤92 elements"
        )

        recommendations.append(
            "Start with smaller datasets to test your system before scaling up"
        )

        return recommendations


def print_alignn_system_info():
    """Print ALIGNN system information to console."""
    checker = ALIGNNSystemInfo()
    checker.check_all()
    print(checker.get_summary())

    recommendations = checker.get_recommendations()
    if recommendations:
        print("\n[ALIGNN Recommendations]")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    print_alignn_system_info()