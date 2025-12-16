"""
ALIGNN Data Preparation and Matching Utilities
Helps users prepare ALIGNN data from separate CIF directory and property file

Features:
- Accept separate CIF directory and property CSV/Excel file
- Automatically match structure files by ID
- Support multiple matching strategies (exact, case-insensitive, fuzzy)
- Generate ALIGNN-compatible dataset
- Validate data integrity
"""

import os
import shutil
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


class DataMatcher:
    """
    Automatically match CIF files with property data and prepare ALIGNN dataset.

    Usage:
        matcher = DataMatcher()
        output_dir, stats = matcher.prepare_dataset(
            cif_dir="path/to/cif_files/",
            property_file="path/to/properties.csv",
            output_dir="path/to/output/",
            id_column="structure_id",
            property_column="band_gap"
        )
    """

    def __init__(self):
        self.cif_dir = None
        self.property_file = None
        self.output_dir = None
        self.matched_data = None
        self.property_df = None

    def scan_cif_directory(self, cif_dir: str) -> Dict[str, Any]:
        """
        Scan CIF directory and return statistics.

        Parameters
        ----------
        cif_dir : str
            Directory containing CIF files

        Returns
        -------
        info : dict
            Directory information including file count and sample names
        """
        if not os.path.exists(cif_dir):
            raise ValueError(f"Directory does not exist: {cif_dir}")

        cif_files = []
        for file in os.listdir(cif_dir):
            if file.lower().endswith('.cif'):
                cif_files.append(file[:-4])  # Remove .cif extension

        return {
            'directory': cif_dir,
            'count': len(cif_files),
            'samples': cif_files[:10],  # First 10 as samples
            'all_files': cif_files
        }

    def load_property_file(self, property_file: str) -> pd.DataFrame:
        """
        Load property file (CSV or Excel).

        Parameters
        ----------
        property_file : str
            Path to CSV or Excel file

        Returns
        -------
        df : pd.DataFrame
            Loaded DataFrame
        """
        if not os.path.exists(property_file):
            raise ValueError(f"File does not exist: {property_file}")

        ext = os.path.splitext(property_file)[1].lower()

        if ext == '.csv':
            # Try different encodings
            for encoding in ['utf-8', 'gbk', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(property_file, encoding=encoding)
                    self.property_df = df
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode CSV file: {property_file}")

        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(property_file)
            self.property_df = df
            return df

        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, or .xls")

    def get_columns(self, property_file: Optional[str] = None) -> List[str]:
        """
        Get column names from property file.

        Parameters
        ----------
        property_file : str, optional
            Path to property file (uses cached if already loaded)

        Returns
        -------
        columns : list
            List of column names
        """
        if property_file:
            self.load_property_file(property_file)

        if self.property_df is None:
            raise ValueError("No property file loaded. Call load_property_file first.")

        return list(self.property_df.columns)

    def detect_id_column(self, df: Optional[pd.DataFrame] = None) -> str:
        """
        Auto-detect ID column from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to analyze (uses cached if not provided)

        Returns
        -------
        id_column : str
            Detected ID column name
        """
        if df is None:
            df = self.property_df

        if df is None:
            raise ValueError("No DataFrame available")

        # Common ID column names (priority order)
        candidates = [
            'id', 'ID', 'Id',
            'structure_id', 'Structure_ID', 'structure_ID',
            'cif_id', 'CIF_ID', 'cif_ID',
            'material_id', 'Material_ID',
            'jid', 'JID',
            'mp_id', 'MP_ID', 'mpid',
            'name', 'Name', 'NAME',
            'formula', 'Formula', 'FORMULA',
            'filename', 'Filename', 'file_name'
        ]

        for col in df.columns:
            if col in candidates:
                return col
            if 'id' in col.lower():
                return col

        # If not found, use first column
        warnings.warn(f"Could not auto-detect ID column, using first column: {df.columns[0]}")
        return df.columns[0]

    def detect_property_columns(self, df: Optional[pd.DataFrame] = None, exclude_id: bool = True) -> List[str]:
        """
        Detect numeric property columns.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to analyze
        exclude_id : bool
            Whether to exclude detected ID column

        Returns
        -------
        property_columns : list
            List of numeric column names
        """
        if df is None:
            df = self.property_df

        if df is None:
            raise ValueError("No DataFrame available")

        id_column = self.detect_id_column(df) if exclude_id else None

        numeric_cols = []
        for col in df.columns:
            if exclude_id and col == id_column:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

        return numeric_cols

    def preview_match(
        self,
        cif_dir: str,
        property_file: str,
        id_column: Optional[str] = None,
        max_preview: int = 20
    ) -> Dict[str, Any]:
        """
        Preview matching results without creating dataset.

        Parameters
        ----------
        cif_dir : str
            Directory containing CIF files
        property_file : str
            Path to property file
        id_column : str, optional
            ID column name (auto-detect if None)
        max_preview : int
            Maximum number of matches to preview

        Returns
        -------
        preview : dict
            Preview of matching results
        """
        # Scan CIF directory
        cif_info = self.scan_cif_directory(cif_dir)
        cif_files = set(cif_info['all_files'])

        # Build lookup indices
        cif_files_lower = {f.lower(): f for f in cif_files}
        cif_files_normalized = {
            f.replace('-', '').replace('_', '').replace(' ', '').lower(): f
            for f in cif_files
        }

        # Load property file
        df = self.load_property_file(property_file)

        # Detect ID column
        if id_column is None:
            id_column = self.detect_id_column(df)

        # Try matching
        matches = []
        unmatched = []

        for idx, row in df.iterrows():
            structure_id = str(row[id_column]).strip()
            matched_id, strategy = self._find_match(
                structure_id, cif_files, cif_files_lower, cif_files_normalized
            )

            if matched_id:
                matches.append({
                    'property_id': structure_id,
                    'cif_id': matched_id,
                    'strategy': strategy
                })
            else:
                unmatched.append(structure_id)

            if len(matches) >= max_preview and len(unmatched) >= 5:
                break

        # Find CIF files without properties
        matched_cif_ids = set(m['cif_id'] for m in matches)
        cif_without_property = [f for f in cif_files if f not in matched_cif_ids][:10]

        return {
            'total_cif_files': len(cif_files),
            'total_property_rows': len(df),
            'id_column': id_column,
            'available_columns': list(df.columns),
            'numeric_columns': self.detect_property_columns(df),
            'matched_preview': matches[:max_preview],
            'unmatched_preview': unmatched[:10],
            'cif_without_property': cif_without_property,
            'estimated_match_rate': len(matches) / min(len(df), max_preview) * 100
        }

    def _find_match(
        self,
        target_id: str,
        cif_files: set,
        cif_files_lower: dict,
        cif_files_normalized: dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find matching CIF file using multiple strategies.

        Returns
        -------
        matched_id : str or None
        strategy : str or None
        """
        # Strategy 1: Exact match
        if target_id in cif_files:
            return target_id, 'exact'

        # Strategy 2: Remove .cif extension
        if target_id.endswith('.cif'):
            alt_id = target_id[:-4]
            if alt_id in cif_files:
                return alt_id, 'remove_ext'

        # Strategy 3: Case-insensitive
        target_lower = target_id.lower()
        if target_lower in cif_files_lower:
            return cif_files_lower[target_lower], 'case_insensitive'

        # Strategy 4: Case-insensitive + remove extension
        if target_lower.endswith('.cif'):
            alt_lower = target_lower[:-4]
            if alt_lower in cif_files_lower:
                return cif_files_lower[alt_lower], 'case_insensitive_no_ext'

        # Strategy 5: Normalized (no separators)
        normalized = target_id.replace('-', '').replace('_', '').replace(' ', '').lower()
        if normalized in cif_files_normalized:
            return cif_files_normalized[normalized], 'normalized'

        # Strategy 6: Fuzzy match
        best_match, best_score = None, 0
        for cif_id in cif_files:
            score = SequenceMatcher(None, target_lower, cif_id.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = cif_id

        if best_score >= 0.90:
            return best_match, 'fuzzy'

        return None, None

    def prepare_dataset(
        self,
        cif_dir: str,
        property_file: str,
        output_dir: str,
        id_column: Optional[str] = None,
        property_column: Optional[str] = None,
        copy_files: bool = True,
        atom_init_file: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare ALIGNN dataset from CIF directory and property file.

        Parameters
        ----------
        cif_dir : str
            Directory containing CIF files
        property_file : str
            CSV/Excel file with structure IDs and properties
        output_dir : str
            Output directory for prepared dataset
        id_column : str, optional
            Name of ID column (auto-detect if None)
        property_column : str, optional
            Name of property column (auto-detect if None)
        copy_files : bool
            Whether to copy CIF files to output directory
        atom_init_file : str, optional
            Custom atom initialization file

        Returns
        -------
        output_dir : str
            Path to prepared dataset
        stats : dict
            Statistics about data preparation
        """
        self.cif_dir = cif_dir
        self.property_file = property_file
        self.output_dir = output_dir

        # Validate inputs
        if not os.path.exists(cif_dir):
            raise ValueError(f"CIF directory does not exist: {cif_dir}")
        if not os.path.exists(property_file):
            raise ValueError(f"Property file does not exist: {property_file}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Scan CIF directory
        print("Scanning CIF directory...")
        cif_info = self.scan_cif_directory(cif_dir)
        cif_files = set(cif_info['all_files'])
        print(f"Found {len(cif_files)} CIF files")

        # Build lookup indices
        cif_files_lower = {f.lower(): f for f in cif_files}
        cif_files_normalized = {
            f.replace('-', '').replace('_', '').replace(' ', '').lower(): f
            for f in cif_files
        }

        # Load property data
        print("Loading property data...")
        df = self.load_property_file(property_file)
        print(f"Loaded {len(df)} rows from property file")

        # Auto-detect columns
        if id_column is None:
            id_column = self.detect_id_column(df)
        if property_column is None:
            numeric_cols = self.detect_property_columns(df)
            if numeric_cols:
                property_column = numeric_cols[0]
            else:
                raise ValueError("No numeric property column found")

        print(f"Using ID column: {id_column}")
        print(f"Using property column: {property_column}")

        # Match data
        print("Matching CIF files with properties...")
        matched = []
        missing_cif = []
        strategy_stats = {}

        for _, row in df.iterrows():
            structure_id = str(row[id_column]).strip()
            property_value = row[property_column]

            # Skip invalid values
            if pd.isna(property_value):
                continue

            matched_id, strategy = self._find_match(
                structure_id, cif_files, cif_files_lower, cif_files_normalized
            )

            if matched_id:
                matched.append((matched_id, float(property_value)))
                strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1
            else:
                missing_cif.append(structure_id)

        # Find CIF files without properties
        matched_ids = set(m[0] for m in matched)
        missing_property = [f for f in cif_files if f not in matched_ids]

        print(f"Matched: {len(matched)} structures")

        # Create id_prop.csv
        print("Creating id_prop.csv...")
        id_prop_file = output_path / 'id_prop.csv'
        with open(id_prop_file, 'w', newline='', encoding='utf-8') as f:
            for structure_id, prop_value in matched:
                f.write(f"{structure_id},{prop_value}\n")

        # Copy CIF files
        if copy_files:
            print("Copying CIF files...")
            cif_source = Path(cif_dir)
            for structure_id, _ in matched:
                src = cif_source / f"{structure_id}.cif"
                dst = output_path / f"{structure_id}.cif"
                if src.exists() and src.resolve() != dst.resolve():
                    shutil.copy(src, dst)

        # Setup atom_init.json
        print("Setting up atom_init.json...")
        self._setup_atom_init(output_path, atom_init_file)

        # Store results
        self.matched_data = matched

        # Compile statistics
        stats = {
            'total_cif_files': len(cif_files),
            'total_property_rows': len(df),
            'matched': len(matched),
            'missing_cif': len(missing_cif),
            'missing_property': len(missing_property),
            'missing_cif_samples': missing_cif[:10],
            'missing_property_samples': missing_property[:10],
            'strategy_stats': strategy_stats,
            'id_column': id_column,
            'property_column': property_column
        }

        print(f"\nDataset prepared successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Total matched: {stats['matched']}")

        return str(output_path), stats

    def _setup_atom_init(self, output_path: Path, custom_file: Optional[str] = None):
        """Setup atom_init.json file."""
        atom_init_file = output_path / 'atom_init.json'

        if custom_file and os.path.exists(custom_file):
            shutil.copy(custom_file, atom_init_file)
            print(f"Using custom atom_init.json")
            return

        # Try to find atom_init.json in module directory
        module_dir = Path(__file__).parent
        default_file = module_dir / 'atom_init.json'

        if default_file.exists():
            shutil.copy(default_file, atom_init_file)
            print("Using default atom_init.json from ALIGNN module")
            return

        # Try CGCNN module
        cgcnn_file = module_dir.parent / 'cgcnn' / 'atom_init.json'
        if cgcnn_file.exists():
            shutil.copy(cgcnn_file, atom_init_file)
            print("Using atom_init.json from CGCNN module")
            return

        # Create default (one-hot encoding)
        print("Creating default atom_init.json (one-hot encoding)")
        atom_init = {}
        for z in range(1, 101):
            features = [0.0] * 92
            if z <= 92:
                features[z - 1] = 1.0
            atom_init[str(z)] = features

        with open(atom_init_file, 'w') as f:
            json.dump(atom_init, f, indent=2)

    def get_property_statistics(self) -> Dict[str, Any]:
        """Get statistics about matched property values."""
        if self.matched_data is None:
            return {}

        values = [v for _, v in self.matched_data]
        if not values:
            return {}

        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }

    def validate_dataset(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Validate prepared ALIGNN dataset.

        Parameters
        ----------
        dataset_dir : str
            Directory to validate

        Returns
        -------
        validation : dict
            Validation results
        """
        issues = []
        dataset_path = Path(dataset_dir)

        if not dataset_path.exists():
            return {'valid': False, 'issues': ['Directory does not exist']}

        # Check id_prop.csv
        id_prop_file = dataset_path / 'id_prop.csv'
        num_entries = 0
        if not id_prop_file.exists():
            issues.append('Missing id_prop.csv')
        else:
            with open(id_prop_file) as f:
                num_entries = sum(1 for _ in f)

        # Check atom_init.json
        atom_init_file = dataset_path / 'atom_init.json'
        if not atom_init_file.exists():
            issues.append('Missing atom_init.json')

        # Check CIF files
        cif_files = list(dataset_path.glob('*.cif'))
        num_cif = len(cif_files)

        if num_entries > 0 and num_entries != num_cif:
            issues.append(
                f'Mismatch: {num_entries} entries in id_prop.csv but {num_cif} CIF files'
            )

        # Verify each entry has corresponding CIF
        if id_prop_file.exists():
            missing_cif = []
            with open(id_prop_file) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if parts:
                        cif_id = parts[0]
                        cif_path = dataset_path / f"{cif_id}.cif"
                        if not cif_path.exists():
                            missing_cif.append(cif_id)

            if missing_cif:
                issues.append(f'{len(missing_cif)} CIF files referenced but not found')

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'num_entries': num_entries,
            'num_cif_files': num_cif,
            'directory': str(dataset_path)
        }


# Convenience function
def prepare_alignn_data(
    cif_dir: str,
    property_file: str,
    output_dir: str,
    id_column: Optional[str] = None,
    property_column: Optional[str] = None,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to prepare ALIGNN dataset.

    Parameters
    ----------
    cif_dir : str
        Directory containing CIF files
    property_file : str
        CSV/Excel file with properties
    output_dir : str
        Output directory
    id_column : str, optional
        ID column name
    property_column : str, optional
        Property column name
    **kwargs
        Additional arguments

    Returns
    -------
    output_dir : str
        Path to prepared dataset
    stats : dict
        Statistics
    """
    matcher = DataMatcher()
    return matcher.prepare_dataset(
        cif_dir=cif_dir,
        property_file=property_file,
        output_dir=output_dir,
        id_column=id_column,
        property_column=property_column,
        **kwargs
    )
