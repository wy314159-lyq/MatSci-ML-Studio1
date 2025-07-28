#!/usr/bin/env python3
"""
MatSci-ML Studio - Main Entry Point
A comprehensive machine learning GUI application for materials science
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ui.main_window import main

if __name__ == "__main__":
    main() 