"""
Data Preparation and Matching Utilities
Helps users prepare CGCNN data from separate CIF directory and property file
"""

import os
import shutil
import json
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import warnings


class DataMatcher:
    """
    Automatically match CIF files with property data and prepare CGCNN dataset.

    Features:
    - Accept separate CIF directory and property CSV file
    - Automatically match by structure ID
    - Generate required CGCNN directory structure
    - Handle various CSV formats
    - Validate data integrity
    """

    def __init__(self):
        self.cif_dir = None
        self.property_file = None
        self.output_dir = None
        self.matched_data = None

    def prepare_dataset(
        self,
        cif_dir: str,
        property_file: str,
        output_dir: str,
        property_column: Optional[str] = None,
        id_column: Optional[str] = None,
        atom_init_file: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Prepare CGCNN dataset from separate CIF directory and property file.

        Parameters
        ----------
        cif_dir : str
            Directory containing CIF files
        property_file : str
            CSV/Excel file with structure IDs and properties
        output_dir : str
            Output directory for CGCNN-formatted data
        property_column : str, optional
            Name of property column (auto-detect if None)
        id_column : str, optional
            Name of ID column (auto-detect if None)
        atom_init_file : str, optional
            Custom atom initialization file (use default if None)

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

        # Load property data
        print("Loading property data...")
        property_df = self._load_property_file(property_file)

        # Auto-detect columns if needed
        if id_column is None:
            id_column = self._detect_id_column(property_df)
        if property_column is None:
            property_column = self._detect_property_column(property_df, id_column)

        print(f"Using ID column: {id_column}")
        print(f"Using property column: {property_column}")

        # Match CIF files with properties
        print("Matching CIF files with properties...")
        matched, stats = self._match_data(
            cif_dir, property_df, id_column, property_column
        )

        # Generate id_prop.csv
        print("Creating id_prop.csv...")
        self._create_id_prop_csv(matched, output_path)

        # Copy CIF files
        print("Copying CIF files...")
        self._copy_cif_files(matched, Path(cif_dir), output_path)

        # Create or copy atom_init.json
        print("Setting up atom_init.json...")
        self._setup_atom_init(output_path, atom_init_file)

        # Store matched data
        self.matched_data = matched

        print(f"\nDataset prepared successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Total structures: {stats['total']}")
        print(f"Matched: {stats['matched']}")
        print(f"Missing CIF: {stats['missing_cif']}")
        print(f"Missing property: {stats['missing_property']}")

        return output_dir, stats

    def _load_property_file(self, file_path: str) -> pd.DataFrame:
        """Load property file (CSV or Excel)."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            # Try different encodings
            for encoding in ['utf-8', 'gbk', 'latin1']:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode CSV file: {file_path}")
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _detect_id_column(self, df: pd.DataFrame) -> str:
        """Auto-detect ID column."""
        # Common ID column names
        candidates = ['id', 'structure_id', 'cif_id', 'name', 'formula',
                     'ID', 'Structure_ID', 'CIF_ID', 'Name', 'Formula']

        for col in df.columns:
            if col in candidates or 'id' in col.lower():
                return col

        # If not found, use first column
        warnings.warn(f"Could not auto-detect ID column, using first column: {df.columns[0]}")
        return df.columns[0]

    def _detect_property_column(self, df: pd.DataFrame, id_column: str) -> str:
        """Auto-detect property column."""
        # Skip ID column and look for numeric column
        for col in df.columns:
            if col != id_column and pd.api.types.is_numeric_dtype(df[col]):
                return col

        # If not found, use second column
        if len(df.columns) > 1:
            warnings.warn(f"Could not auto-detect property column, using: {df.columns[1]}")
            return df.columns[1]
        else:
            raise ValueError("Property file must have at least 2 columns")

    def _match_data(
        self,
        cif_dir: str,
        property_df: pd.DataFrame,
        id_column: str,
        property_column: str
    ) -> Tuple[List[Tuple[str, float]], dict]:
        """
        Match CIF files with properties using multiple strategies.

        Matching strategies (in order):
        1. Exact match (case-sensitive)
        2. Case-insensitive match
        3. Remove .cif extension from ID
        4. Remove common separators (-, _, space)
        5. Fuzzy match (similarity > 90%)
        """
        # Get list of CIF files with multiple index formats
        cif_files = {}
        cif_files_lower = {}  # lowercase mapping
        cif_files_normalized = {}  # normalized mapping (no separators)

        for file in os.listdir(cif_dir):
            if file.endswith('.cif'):
                # Remove .cif extension to get ID
                cif_id = file[:-4]
                cif_files[cif_id] = cif_id

                # Lowercase version
                cif_files_lower[cif_id.lower()] = cif_id

                # Normalized version (remove separators)
                normalized = cif_id.replace('-', '').replace('_', '').replace(' ', '').lower()
                cif_files_normalized[normalized] = cif_id

        # Match with properties
        matched = []
        missing_cif = []
        match_details = []  # Track which strategy worked

        for _, row in property_df.iterrows():
            structure_id = str(row[id_column]).strip()
            property_value = row[property_column]

            matched_id = None
            match_strategy = None

            # Strategy 1: Exact match (case-sensitive)
            if structure_id in cif_files:
                matched_id = structure_id
                match_strategy = 'exact'

            # Strategy 2: Try with .cif extension removed
            elif structure_id.endswith('.cif'):
                alt_id = structure_id[:-4]
                if alt_id in cif_files:
                    matched_id = alt_id
                    match_strategy = 'remove_cif_ext'

            # Strategy 3: Case-insensitive match
            if not matched_id:
                structure_id_lower = structure_id.lower()
                if structure_id_lower in cif_files_lower:
                    matched_id = cif_files_lower[structure_id_lower]
                    match_strategy = 'case_insensitive'

            # Strategy 4: Case-insensitive with .cif removed
            if not matched_id and structure_id.lower().endswith('.cif'):
                alt_id_lower = structure_id.lower()[:-4]
                if alt_id_lower in cif_files_lower:
                    matched_id = cif_files_lower[alt_id_lower]
                    match_strategy = 'case_insensitive_no_ext'

            # Strategy 5: Normalized match (no separators)
            if not matched_id:
                normalized_id = structure_id.replace('-', '').replace('_', '').replace(' ', '').lower()
                if normalized_id in cif_files_normalized:
                    matched_id = cif_files_normalized[normalized_id]
                    match_strategy = 'normalized'

            # Strategy 6: Fuzzy match (similarity-based)
            if not matched_id:
                matched_id, match_strategy = self._fuzzy_match(
                    structure_id, list(cif_files.keys())
                )

            # Record result
            if matched_id:
                matched.append((matched_id, float(property_value)))
                match_details.append({
                    'original_id': structure_id,
                    'matched_id': matched_id,
                    'strategy': match_strategy
                })
            else:
                missing_cif.append(structure_id)

        # Check for CIF files without properties
        matched_ids = set(m[0] for m in matched)
        missing_property = []
        for cif_id in cif_files:
            if cif_id not in matched_ids:
                missing_property.append(cif_id)

        # Calculate match strategy statistics
        strategy_stats = {}
        for detail in match_details:
            strategy = detail['strategy']
            strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1

        stats = {
            'total': len(property_df),
            'matched': len(matched),
            'missing_cif': len(missing_cif),
            'missing_property': len(missing_property),
            'missing_cif_list': missing_cif[:10],  # Show first 10
            'missing_property_list': missing_property[:10],
            'match_details': match_details[:20],  # Show first 20 matches with strategies
            'strategy_stats': strategy_stats  # Statistics of which strategies were used
        }

        if missing_cif:
            warnings.warn(f"{len(missing_cif)} structures in property file have no CIF file")
        if missing_property:
            warnings.warn(f"{len(missing_property)} CIF files have no property value")

        return matched, stats

    def _fuzzy_match(self, target: str, candidates: List[str], threshold: float = 0.90) -> Tuple[Optional[str], Optional[str]]:
        """
        Fuzzy match using string similarity.

        Parameters
        ----------
        target : str
            The ID to match
        candidates : List[str]
            List of candidate CIF IDs
        threshold : float
            Minimum similarity score (0-1, default: 0.90)

        Returns
        -------
        matched_id : str or None
            Best matching candidate ID, or None if no good match
        strategy : str or None
            'fuzzy' if matched, None otherwise
        """
        try:
            from difflib import SequenceMatcher
        except ImportError:
            return None, None

        best_match = None
        best_score = 0

        target_lower = target.lower()

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Calculate similarity
            similarity = SequenceMatcher(None, target_lower, candidate_lower).ratio()

            if similarity > best_score:
                best_score = similarity
                best_match = candidate

        # Only return match if similarity is high enough
        if best_score >= threshold:
            return best_match, 'fuzzy'
        else:
            return None, None

    def _create_id_prop_csv(self, matched: List[Tuple[str, float]], output_path: Path):
        """Create id_prop.csv file."""
        id_prop_file = output_path / 'id_prop.csv'
        with open(id_prop_file, 'w', newline='') as f:
            for structure_id, property_value in matched:
                f.write(f"{structure_id},{property_value}\n")

    def _copy_cif_files(self, matched: List[Tuple[str, float]], cif_dir: Path, output_path: Path):
        """Copy matched CIF files to output directory."""
        for structure_id, _ in matched:
            src_file = cif_dir / f"{structure_id}.cif"
            dst_file = output_path / f"{structure_id}.cif"

            if src_file.exists():
                # Check if source and destination are the same
                if src_file.resolve() == dst_file.resolve():
                    print(f"Skipping copy: {structure_id}.cif (source and destination are the same)")
                    continue

                shutil.copy(src_file, dst_file)
            else:
                warnings.warn(f"CIF file not found: {src_file}")

    def _setup_atom_init(self, output_path: Path, custom_file: Optional[str] = None):
        """Setup atom_init.json file.

        Priority:
        1. Custom file if provided and exists
        2. Official atom_init.json from CGCNN module directory (92-dim features for elements 1-100)
        3. Fallback to simple one-hot encoding (not recommended)
        """
        atom_init_file = output_path / 'atom_init.json'

        if custom_file and os.path.exists(custom_file):
            # Copy custom file
            shutil.copy(custom_file, atom_init_file)
            print(f"Using custom atom_init.json from: {custom_file}")
        else:
            # Try to use official atom_init.json from CGCNN module directory
            official_atom_init = self._get_official_atom_init()
            if official_atom_init and os.path.exists(official_atom_init):
                shutil.copy(official_atom_init, atom_init_file)
                print(f"Using official atom_init.json (92-dim features for elements 1-100)")
            else:
                # Fallback: Create default atom initialization (not recommended)
                self._create_default_atom_init(atom_init_file)
                print("WARNING: Created simple one-hot atom_init.json (not official)")
                print("For best results, use official atom_init.json from CGCNN module")

    def _get_official_atom_init(self) -> Optional[str]:
        """Get path to official atom_init.json in CGCNN module directory.

        Returns the path to the official 92-dimensional atom feature vectors
        for elements 1-100, based on the original CGCNN implementation.
        """
        # Get the directory where this module is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        official_path = os.path.join(module_dir, 'atom_init.json')

        if os.path.exists(official_path):
            return official_path

        return None

    def _create_default_atom_init(self, output_file: Path):
        """Create default atom initialization file for elements 1-100.

        WARNING: This creates simple one-hot encoding which is NOT the same as
        the official CGCNN atom features (92-dim). For best results, use the official
        atom_init.json from the CGCNN module directory.

        This fallback creates 92-dim one-hot vectors for elements 1-100 to match
        the official format dimensions.
        """
        # Simple one-hot encoding for elements 1-100 with 92-dim vectors
        # (matches official atom_init.json dimensions)
        atom_init = {}
        num_elements = 100  # Elements 1-100 (same as official)
        feature_dim = 92    # Same dimension as official

        for elem in range(1, num_elements + 1):
            features = [0.0] * feature_dim
            if elem <= feature_dim:
                features[elem - 1] = 1.0
            atom_init[str(elem)] = features

        with open(output_file, 'w') as f:
            json.dump(atom_init, f, indent=2)

    def get_statistics(self) -> dict:
        """Get statistics about matched data."""
        if self.matched_data is None:
            return {}

        values = [v for _, v in self.matched_data]
        return {
            'count': len(values),
            'mean': sum(values) / len(values) if values else 0,
            'min': min(values) if values else 0,
            'max': max(values) if values else 0,
            'std': pd.Series(values).std() if values else 0
        }

    def validate_dataset(self, dataset_dir: str) -> dict:
        """
        Validate a CGCNN dataset directory.

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

        # Check directory exists
        if not os.path.exists(dataset_dir):
            return {'valid': False, 'issues': ['Directory does not exist']}

        # Check id_prop.csv
        id_prop_file = os.path.join(dataset_dir, 'id_prop.csv')
        if not os.path.exists(id_prop_file):
            issues.append('Missing id_prop.csv')
        else:
            # Count lines
            with open(id_prop_file) as f:
                num_structures = len(f.readlines())

        # Check atom_init.json
        atom_init_file = os.path.join(dataset_dir, 'atom_init.json')
        if not os.path.exists(atom_init_file):
            issues.append('Missing atom_init.json')

        # Check CIF files
        cif_files = [f for f in os.listdir(dataset_dir) if f.endswith('.cif')]
        num_cif = len(cif_files)

        # Compare counts
        if num_structures != num_cif:
            issues.append(f'Mismatch: {num_structures} entries in id_prop.csv but {num_cif} CIF files')

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'num_structures': num_structures if 'num_structures' in locals() else 0,
            'num_cif_files': num_cif
        }


# Convenience function
def prepare_cgcnn_data(
    cif_dir: str,
    property_file: str,
    output_dir: str,
    **kwargs
) -> Tuple[str, dict]:
    """
    Convenience function to prepare CGCNN dataset.

    Parameters
    ----------
    cif_dir : str
        Directory containing CIF files
    property_file : str
        CSV/Excel file with properties
    output_dir : str
        Output directory
    **kwargs
        Additional arguments passed to DataMatcher.prepare_dataset

    Returns
    -------
    output_dir : str
        Path to prepared dataset
    stats : dict
        Statistics
    """
    matcher = DataMatcher()
    return matcher.prepare_dataset(cif_dir, property_file, output_dir, **kwargs)
