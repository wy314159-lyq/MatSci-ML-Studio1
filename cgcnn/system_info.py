"""
System Information Checker for CGCNN
Checks hardware and software environment requirements
"""

import platform
import sys
import os
import subprocess
from typing import Dict, List, Tuple


class SystemInfo:
    """
    System information checker for CGCNN training requirements.

    Checks:
    - Python version
    - PyTorch installation and CUDA availability
    - System resources (CPU, RAM, GPU)
    - Dependencies
    - Disk space
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
                f"Python {version_str} detected. Python 3.7+ is recommended."
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

                # Check GPU memory
                min_memory_gb = 4
                for gpu in gpu_info:
                    if float(gpu['memory_gb']) < min_memory_gb:
                        self.warnings.append(
                            f"GPU {gpu['id']} ({gpu['name']}) has only "
                            f"{gpu['memory_gb']}GB memory. "
                            f"At least {min_memory_gb}GB recommended for training."
                        )
            else:
                self.info['cuda_available'] = False
                self.info['gpu_count'] = 0
                self.warnings.append(
                    "CUDA not available. Training will use CPU (slower)."
                )

        except ImportError:
            self.info['cuda_available'] = False
            self.info['gpu_count'] = 0

    def check_dependencies(self):
        """Check required dependencies."""
        dependencies = {
            # Core dependencies
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'scipy': 'SciPy',
            'pandas': 'Pandas',

            # Materials science
            'pymatgen': 'Pymatgen',
            'jarvis': 'JARVIS-Tools',
            'ase': 'ASE (Atomic Simulation Environment)',

            # Deep learning
            'dgl': 'DGL (Deep Graph Library)',
            'sklearn': 'Scikit-learn',

            # UI
            'PyQt5': 'PyQt5 (for UI)',

            # Visualization
            'matplotlib': 'Matplotlib',
            'seaborn': 'Seaborn',

            # Data processing
            'tqdm': 'tqdm (progress bars)',
            'pydantic': 'Pydantic (data validation)',

            # Optimization
            'pymoo': 'pymoo (multi-objective optimization)',

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

        # Check critical dependencies
        critical = ['PyTorch', 'NumPy', 'Pymatgen']
        for dep in critical:
            if dep in missing:
                self.errors.append(f"Critical dependency missing: {dep}")

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
        import psutil

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

        # Check RAM requirements
        min_ram_gb = 8
        total_ram_gb = memory.total / (1024**3)
        if total_ram_gb < min_ram_gb:
            self.warnings.append(
                f"System has {total_ram_gb:.1f}GB RAM. "
                f"At least {min_ram_gb}GB recommended for training."
            )

        # Check available RAM
        if memory.percent > 80:
            self.warnings.append(
                f"RAM usage is high ({memory.percent}%). "
                f"Close other applications before training."
            )

    def check_disk_space(self):
        """Check available disk space."""
        try:
            import psutil

            # Get disk usage for current directory
            # Use root directory to avoid path issues on Windows
            if platform.system() == 'Windows':
                # Use C: drive on Windows
                disk_path = 'C:\\'
            else:
                disk_path = '/'

            disk = psutil.disk_usage(disk_path)
            self.info['disk_total_gb'] = f"{disk.total / (1024**3):.2f}"
            self.info['disk_free_gb'] = f"{disk.free / (1024**3):.2f}"
            self.info['disk_percent_used'] = disk.percent

            # Check disk space
            min_free_gb = 10
            free_gb = disk.free / (1024**3)
            if free_gb < min_free_gb:
                self.warnings.append(
                    f"Low disk space: {free_gb:.1f}GB free. "
                    f"At least {min_free_gb}GB recommended."
                )
        except Exception as e:
            # If disk check fails, just skip it
            self.info['disk_total_gb'] = "N/A"
            self.info['disk_free_gb'] = "N/A"
            self.info['disk_percent_used'] = "N/A"


    def get_summary(self) -> str:
        """
        Get formatted summary of system information.

        Returns
        -------
        str
            Formatted system information
        """
        lines = []
        lines.append("=" * 70)
        lines.append("System Information & Environment Check")
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
                # Show both PyTorch's CUDA version and system CUDA version
                pytorch_cuda = self.info.get('cuda_version_pytorch', 'Unknown')
                system_cuda = self.info.get('cuda_version_system', 'Not detected')

                lines.append(f"  CUDA Version (PyTorch): {pytorch_cuda}")
                lines.append(f"  CUDA Version (System): {system_cuda}")
                lines.append(f"  cuDNN Version: {self.info.get('cudnn_version', 'Unknown')}")

                # Add warning if versions don't match
                if (system_cuda != 'Not detected' and
                    system_cuda != 'Unknown (nvcc not found or failed)' and
                    pytorch_cuda != system_cuda):
                    self.warnings.append(
                        f"CUDA version mismatch: PyTorch compiled with {pytorch_cuda}, "
                        f"but system has {system_cuda}. This may cause compatibility issues."
                    )
        else:
            lines.append("  NOT INSTALLED")

        # GPU
        lines.append("\n[GPU]")
        gpu_count = self.info.get('gpu_count', 0)
        lines.append(f"  Count: {gpu_count}")
        if gpu_count > 0:
            for gpu in self.info.get('gpus', []):
                lines.append(f"  GPU {gpu['id']}: {gpu['name']}")
                lines.append(f"    Memory: {gpu['memory_gb']} GB")
        else:
            lines.append("  No GPU detected (CPU mode only)")

        # CPU & RAM
        lines.append("\n[System Resources]")
        lines.append(f"  CPU Cores (Physical): {self.info.get('cpu_count', 'Unknown')}")
        lines.append(f"  CPU Cores (Logical): {self.info.get('cpu_count_logical', 'Unknown')}")
        if 'cpu_freq_mhz' in self.info:
            lines.append(f"  CPU Frequency: {self.info['cpu_freq_mhz']} MHz")
        lines.append(f"  RAM Total: {self.info.get('ram_total_gb', 'Unknown')} GB")
        lines.append(f"  RAM Available: {self.info.get('ram_available_gb', 'Unknown')} GB")
        lines.append(f"  RAM Usage: {self.info.get('ram_percent_used', 'Unknown')}%")

        # Disk
        lines.append("\n[Disk Space]")
        lines.append(f"  Total: {self.info.get('disk_total_gb', 'Unknown')} GB")
        lines.append(f"  Free: {self.info.get('disk_free_gb', 'Unknown')} GB")
        lines.append(f"  Usage: {self.info.get('disk_percent_used', 'Unknown')}%")

        # Dependencies - Core
        lines.append("\n[Core Dependencies]")
        installed = self.info.get('dependencies_installed', {})
        core_deps = ['PyTorch', 'NumPy', 'SciPy', 'Pandas']
        for dep in core_deps:
            if dep in installed:
                lines.append(f"  [OK] {dep}: {installed[dep]}")
            else:
                lines.append(f"  [MISSING] {dep}")

        # Dependencies - Materials Science
        lines.append("\n[Materials Science Libraries]")
        mat_deps = ['Pymatgen', 'JARVIS-Tools', 'ASE (Atomic Simulation Environment)']
        for dep in mat_deps:
            if dep in installed:
                lines.append(f"  [OK] {dep}: {installed[dep]}")
            else:
                lines.append(f"  [MISSING] {dep}")

        # Dependencies - Deep Learning
        lines.append("\n[Deep Learning Libraries]")
        dl_deps = ['DGL (Deep Graph Library)', 'Scikit-learn']
        for dep in dl_deps:
            if dep in installed:
                lines.append(f"  [OK] {dep}: {installed[dep]}")
            else:
                lines.append(f"  [MISSING] {dep}")

        # Dependencies - UI & Visualization
        lines.append("\n[UI & Visualization]")
        ui_deps = ['PyQt5 (for UI)', 'Matplotlib', 'Seaborn']
        for dep in ui_deps:
            if dep in installed:
                lines.append(f"  [OK] {dep}: {installed[dep]}")
            else:
                lines.append(f"  [MISSING] {dep}")

        # Dependencies - Others
        lines.append("\n[Other Libraries]")
        other_deps = ['tqdm (progress bars)', 'Pydantic (data validation)',
                      'pymoo (multi-objective optimization)', 'psutil (system monitoring)',
                      'openpyxl (Excel support)']
        for dep in other_deps:
            if dep in installed:
                lines.append(f"  [OK] {dep}: {installed[dep]}")
            else:
                lines.append(f"  [MISSING] {dep}")

        # All missing dependencies summary
        missing = self.info.get('dependencies_missing', [])
        if missing:
            lines.append("\n[Missing Dependencies Summary]")
            lines.append(f"  Total missing: {len(missing)}")
            for name in missing:
                lines.append(f"    - {name}")

        # Environment Variables
        lines.append("\n[Environment Variables]")
        env_vars = self.info.get('environment_variables', {})
        for key, value in env_vars.items():
            if key == 'PATH':
                lines.append(f"  {key}: (see below)")
            else:
                lines.append(f"  {key}: {value}")

        # Python Path (first 5 entries)
        lines.append("\n[Python Path (first 5 entries)]")
        python_path = self.info.get('python_path', [])
        for i, path in enumerate(python_path[:5], 1):
            lines.append(f"  {i}. {path}")
        if len(python_path) > 5:
            lines.append(f"  ... and {len(python_path) - 5} more entries")

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
            lines.append("Action: Please fix errors before training")
        elif self.warnings:
            lines.append("STATUS: WARNINGS PRESENT")
            lines.append("Action: System may work but performance may be limited")
        else:
            lines.append("STATUS: ALL CHECKS PASSED")
            lines.append("Action: System ready for training")
        lines.append("=" * 70)

        return "\n".join(lines)

    def get_recommendations(self) -> List[str]:
        """
        Get system improvement recommendations.

        Returns
        -------
        list
            List of recommendations
        """
        recommendations = []

        # GPU recommendation
        if not self.info.get('cuda_available'):
            recommendations.append(
                "Consider using a system with NVIDIA GPU for faster training. "
                "CPU training is much slower for neural networks."
            )

        # RAM recommendation
        try:
            ram_gb = float(self.info.get('ram_total_gb', 0))
            if ram_gb < 16:
                recommendations.append(
                    f"System has {ram_gb:.1f}GB RAM. "
                    f"Consider 16GB+ for better performance with large datasets."
                )
        except ValueError:
            pass  # Skip if RAM info not available

        # Disk space recommendation
        try:
            free_gb = float(self.info.get('disk_free_gb', 0))
            if free_gb < 50:
                recommendations.append(
                    f"Free disk space: {free_gb:.1f}GB. "
                    f"Consider having 50GB+ free for datasets and checkpoints."
                )
        except ValueError:
            pass  # Skip if disk info not available

        # Dependencies
        missing = self.info.get('dependencies_missing', [])
        if missing:
            recommendations.append(
                f"Install missing dependencies: pip install {' '.join(missing)}"
            )

        return recommendations


def print_system_info():
    """Print system information to console."""
    checker = SystemInfo()
    checker.check_all()
    print(checker.get_summary())

    recommendations = checker.get_recommendations()
    if recommendations:
        print("\n[Recommendations]")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    print_system_info()
