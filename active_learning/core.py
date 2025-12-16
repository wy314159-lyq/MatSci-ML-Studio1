"""
Active Learning Core Module

This module implements the core components for active learning-based
experimental design and feature combination search.

Components:
- DataStore: Manages labeled experiment data
- DesignSpace: Defines and generates candidate feature combinations
- BaseModelWrapper / EnsembleModel: Model abstraction and ensemble predictions
- CandidateEvaluator: Scores candidates using various acquisition strategies
- Selector: Selects top candidates while avoiding duplicates
- ExperimentInterface: Abstract interface for running real experiments
- ActiveLoop: Orchestrates the complete active learning iteration
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
import numpy as np


# =============================================================================
# 1. DataStore: Experiment Data Storage and Management
# =============================================================================

class DataStore:
    """
    Manages storage and updates of labeled experiment data.

    Maintains the current set of experiments that have been performed,
    including their feature configurations (X) and observed results (y).

    Attributes:
        X_labeled (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y_labeled (np.ndarray): Target values of shape (n_samples,)
        feature_names (List[str]): Names of features
    """

    def __init__(self, X_init: Optional[np.ndarray] = None,
                 y_init: Optional[np.ndarray] = None,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize the DataStore with optional initial data.

        Args:
            X_init: Initial feature matrix, shape (n_samples, n_features)
            y_init: Initial target values, shape (n_samples,)
            feature_names: Names of features
        """
        if X_init is not None and y_init is not None:
            self.X_labeled = np.array(X_init, dtype=np.float64)
            self.y_labeled = np.array(y_init, dtype=np.float64).flatten()
        else:
            self.X_labeled = np.array([]).reshape(0, 0)
            self.y_labeled = np.array([])

        self.feature_names = feature_names or []

    def get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current labeled dataset.

        Returns:
            Tuple of (X_labeled, y_labeled)
        """
        return self.X_labeled.copy(), self.y_labeled.copy()

    def add_observations(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Add new experiment observations to the dataset.

        Args:
            X_new: New feature matrix, shape (n_new, n_features)
            y_new: New target values, shape (n_new,)
        """
        X_new = np.array(X_new, dtype=np.float64)
        y_new = np.array(y_new, dtype=np.float64).flatten()

        if len(X_new.shape) == 1:
            X_new = X_new.reshape(1, -1)

        if self.X_labeled.size == 0:
            self.X_labeled = X_new
            self.y_labeled = y_new
        else:
            self.X_labeled = np.vstack([self.X_labeled, X_new])
            self.y_labeled = np.concatenate([self.y_labeled, y_new])

    def reset(self) -> None:
        """Reset the data store to empty state."""
        self.X_labeled = np.array([]).reshape(0, 0)
        self.y_labeled = np.array([])

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.y_labeled)

    @property
    def n_features(self) -> int:
        """Number of features."""
        if self.X_labeled.size == 0:
            return 0
        return self.X_labeled.shape[1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'X_labeled': self.X_labeled.tolist() if self.X_labeled.size > 0 else [],
            'y_labeled': self.y_labeled.tolist() if self.y_labeled.size > 0 else [],
            'feature_names': self.feature_names
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataStore':
        """Create from dictionary."""
        X = np.array(data.get('X_labeled', [])) if data.get('X_labeled') else None
        y = np.array(data.get('y_labeled', [])) if data.get('y_labeled') else None
        feature_names = data.get('feature_names', [])
        return cls(X, y, feature_names)


# =============================================================================
# 2. DesignSpace: Feature Design Space and Candidate Generation
# =============================================================================

class FeatureDimension:
    """
    Represents a single feature dimension in the design space.

    Supports both continuous (bounded interval) and discrete (enumerated values)
    feature types.
    """

    def __init__(self, name: str, bounds: Union[Tuple[float, float, str], List],
                 dtype: str = 'float'):
        """
        Initialize a feature dimension.

        Args:
            name: Name of the feature
            bounds: Either (min, max, "continuous") for continuous features,
                   or a list of discrete values [v1, v2, ...]
            dtype: Data type ('float' or 'int')
        """
        self.name = name
        self.dtype = dtype

        if isinstance(bounds, tuple) and len(bounds) == 3 and bounds[2] == "continuous":
            self.type = "continuous"
            self.min_val = float(bounds[0])
            self.max_val = float(bounds[1])
            self.values = None
        elif isinstance(bounds, (list, np.ndarray)):
            self.type = "discrete"
            self.values = list(bounds)
            self.min_val = min(self.values) if self.values else 0
            self.max_val = max(self.values) if self.values else 0
        else:
            raise ValueError(f"Invalid bounds specification for feature '{name}': {bounds}")

    def sample_random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate random samples from this dimension."""
        if self.type == "continuous":
            samples = rng.uniform(self.min_val, self.max_val, size=n)
            if self.dtype == 'int':
                samples = np.round(samples).astype(int)
            return samples
        else:
            return rng.choice(self.values, size=n)

    def get_grid(self, n_points: int) -> np.ndarray:
        """Get grid points for this dimension."""
        if self.type == "continuous":
            grid = np.linspace(self.min_val, self.max_val, n_points)
            if self.dtype == 'int':
                grid = np.unique(np.round(grid).astype(int))
            return grid
        else:
            return np.array(self.values)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        if self.type == "continuous":
            bounds = (self.min_val, self.max_val, "continuous")
        else:
            bounds = self.values
        return {
            'name': self.name,
            'bounds': bounds,
            'dtype': self.dtype,
            'type': self.type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDimension':
        """Create from dictionary."""
        return cls(data['name'], data['bounds'], data.get('dtype', 'float'))


class DesignSpace:
    """
    Defines the feature design space and generates candidate combinations.

    Manages multiple feature dimensions and provides methods to generate
    candidate points through random sampling or grid-based approaches.
    """

    def __init__(self, dimensions: Optional[List[Tuple[str, Union[Tuple, List]]]] = None):
        """
        Initialize the design space.

        Args:
            dimensions: List of (name, bounds) tuples defining each feature.
                       For continuous: ("feature_name", (min, max, "continuous"))
                       For discrete: ("feature_name", [v1, v2, v3, ...])
        """
        self.dimensions: List[FeatureDimension] = []
        if dimensions:
            for item in dimensions:
                if len(item) == 2:
                    name, bounds = item
                    dtype = 'float'
                else:
                    name, bounds, dtype = item
                self.dimensions.append(FeatureDimension(name, bounds, dtype))

    def add_dimension(self, name: str, bounds: Union[Tuple, List],
                      dtype: str = 'float') -> None:
        """Add a new dimension to the design space."""
        self.dimensions.append(FeatureDimension(name, bounds, dtype))

    def clear_dimensions(self) -> None:
        """Clear all dimensions."""
        self.dimensions.clear()

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.dimensions)

    @property
    def feature_names(self) -> List[str]:
        """Names of all features."""
        return [d.name for d in self.dimensions]

    def generate_candidates(self, n_candidates: int,
                           mode: str = "random",
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Generate candidate feature combinations.

        Args:
            n_candidates: Number of candidates to generate
            mode: Generation mode - "random" for random sampling,
                 "grid" for grid-based, "lhs" for Latin Hypercube
            seed: Random seed for reproducibility

        Returns:
            Candidate matrix of shape (n_candidates, n_features)
        """
        if not self.dimensions:
            raise ValueError("No dimensions defined in design space")

        rng = np.random.default_rng(seed)

        if mode == "random":
            return self._generate_random(n_candidates, rng)
        elif mode == "grid":
            return self._generate_grid(n_candidates)
        elif mode == "lhs":
            return self._generate_lhs(n_candidates, rng)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

    def _generate_random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n random candidates."""
        samples = np.zeros((n, self.n_features))
        for i, dim in enumerate(self.dimensions):
            samples[:, i] = dim.sample_random(n, rng)
        return samples

    def _generate_grid(self, n_total: int) -> np.ndarray:
        """Generate grid-based candidates."""
        n_per_dim = max(2, int(np.ceil(n_total ** (1.0 / self.n_features))))
        grids = [dim.get_grid(n_per_dim) for dim in self.dimensions]
        meshes = np.meshgrid(*grids, indexing='ij')
        candidates = np.stack([m.flatten() for m in meshes], axis=1)

        if len(candidates) > n_total:
            indices = np.random.choice(len(candidates), size=n_total, replace=False)
            candidates = candidates[indices]

        return candidates

    def _generate_lhs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate Latin Hypercube Sampling candidates."""
        samples = np.zeros((n, self.n_features))

        for i, dim in enumerate(self.dimensions):
            if dim.type == "continuous":
                # LHS for continuous
                intervals = np.linspace(0, 1, n + 1)
                points = rng.uniform(intervals[:-1], intervals[1:])
                rng.shuffle(points)
                samples[:, i] = dim.min_val + points * (dim.max_val - dim.min_val)
                if dim.dtype == 'int':
                    samples[:, i] = np.round(samples[:, i])
            else:
                # Random for discrete
                samples[:, i] = dim.sample_random(n, rng)

        return samples

    def is_valid(self, X: np.ndarray) -> np.ndarray:
        """Check which candidates are within the valid design space."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        valid = np.ones(len(X), dtype=bool)
        for i, dim in enumerate(self.dimensions):
            if dim.type == "continuous":
                valid &= (X[:, i] >= dim.min_val) & (X[:, i] <= dim.max_val)
            else:
                valid &= np.isin(X[:, i], dim.values)

        return valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dimensions': [dim.to_dict() for dim in self.dimensions]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DesignSpace':
        """Create from dictionary."""
        space = cls()
        for dim_data in data.get('dimensions', []):
            space.dimensions.append(FeatureDimension.from_dict(dim_data))
        return space


# =============================================================================
# 3. Model Wrappers: BaseModelWrapper and EnsembleModel
# =============================================================================

class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.

    Provides a unified interface for different regression models.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass


class SklearnRegressorWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn regression models."""

    def __init__(self, model):
        """
        Initialize the wrapper with a sklearn model.

        Args:
            model: A scikit-learn regressor instance
        """
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the sklearn model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the sklearn model."""
        return self.model.predict(X)


class EnsembleModel:
    """
    Ensemble of models trained via bootstrap sampling.

    Creates multiple base models to enable uncertainty estimation
    through prediction variance.
    """

    def __init__(self,
                 base_model_factory: Callable[[], BaseModelWrapper],
                 n_estimators: int = 10):
        """
        Initialize the ensemble.

        Args:
            base_model_factory: Callable that returns a new BaseModelWrapper instance
            n_estimators: Number of models in the ensemble
        """
        self.base_model_factory = base_model_factory
        self.n_estimators = n_estimators
        self.models: List[BaseModelWrapper] = []

    def fit_ensemble(self, X: np.ndarray, y: np.ndarray,
                     seed: Optional[int] = None) -> None:
        """
        Train the ensemble on data using bootstrap sampling.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        n_samples = len(y)

        self.models = []
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Create and train a new model
            model = self.base_model_factory()
            model.fit(X_boot, y_boot)
            self.models.append(model)

    def predict_all(self, X_candidate: np.ndarray) -> np.ndarray:
        """
        Get predictions from all models in the ensemble.

        Returns:
            Prediction matrix, shape (n_estimators, n_candidates)
        """
        if len(X_candidate.shape) == 1:
            X_candidate = X_candidate.reshape(1, -1)

        predictions = np.array([model.predict(X_candidate) for model in self.models])
        return predictions

    def predict_stats(self, X_candidate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction mean and standard deviation across the ensemble.

        Returns:
            Tuple of (mean, std) where each has shape (n_candidates,)
        """
        predictions = self.predict_all(X_candidate)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std


# =============================================================================
# 4. CandidateEvaluator: Acquisition Function / Scoring Strategy
# =============================================================================

class CandidateEvaluator:
    """
    Evaluates candidates using various acquisition strategies.

    Supported modes:
        - "mean": Pure exploitation (score = mean)
        - "std": Pure exploration (score = std)
        - "ucb": Upper Confidence Bound (score = mean + lambda * std)
        - "ei": Expected Improvement
        - "pi": Probability of Improvement
    """

    MODES = ["mean", "std", "ucb", "ei", "pi"]

    def __init__(self, mode: str = "ucb", ucb_lambda: float = 1.0,
                 maximize: bool = True):
        """
        Initialize the evaluator.

        Args:
            mode: Evaluation strategy
            ucb_lambda: Weight for uncertainty in UCB mode
            maximize: If True, maximize target; if False, minimize
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.MODES}")

        self.mode = mode
        self.ucb_lambda = ucb_lambda
        self.maximize = maximize
        self.best_y: Optional[float] = None

    def set_best_y(self, best_y: float) -> None:
        """Set the current best observed value (needed for EI and PI)."""
        self.best_y = best_y

    def evaluate(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Compute acquisition scores for candidates.

        Args:
            mean: Predicted means, shape (n_candidates,)
            std: Predicted standard deviations, shape (n_candidates,)

        Returns:
            Scores array, shape (n_candidates,). Higher score = more promising.
        """
        if self.mode == "mean":
            scores = mean if self.maximize else -mean
        elif self.mode == "std":
            scores = std
        elif self.mode == "ucb":
            if self.maximize:
                scores = mean + self.ucb_lambda * std
            else:
                scores = -mean + self.ucb_lambda * std
        elif self.mode == "ei":
            scores = self._expected_improvement(mean, std)
        elif self.mode == "pi":
            scores = self._probability_of_improvement(mean, std)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return scores

    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Compute Expected Improvement acquisition function."""
        from scipy.stats import norm

        if self.best_y is None:
            return mean if self.maximize else -mean

        # Avoid division by zero
        std = np.maximum(std, 1e-9)

        if self.maximize:
            z = (mean - self.best_y) / std
            ei = (mean - self.best_y) * norm.cdf(z) + std * norm.pdf(z)
        else:
            z = (self.best_y - mean) / std
            ei = (self.best_y - mean) * norm.cdf(z) + std * norm.pdf(z)

        return ei

    def _probability_of_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Compute Probability of Improvement acquisition function."""
        from scipy.stats import norm

        if self.best_y is None:
            return mean if self.maximize else -mean

        # Avoid division by zero
        std = np.maximum(std, 1e-9)

        if self.maximize:
            z = (mean - self.best_y) / std
        else:
            z = (self.best_y - mean) / std

        return norm.cdf(z)


# =============================================================================
# 5. Selector: Select Top Candidates for Next Experiments
# =============================================================================

class Selector:
    """
    Selects top-k candidates based on scores while avoiding duplicates.
    """

    def __init__(self, tolerance: float = 1e-8):
        """
        Initialize the selector.

        Args:
            tolerance: Tolerance for floating-point comparison
        """
        self.tolerance = tolerance

    def select(self,
               X_candidate: np.ndarray,
               scores: np.ndarray,
               k: int,
               X_existing: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-k candidates by score, excluding existing points.

        Returns:
            Tuple of (selected_candidates, selected_scores)
        """
        # Sort candidates by score (descending)
        sorted_indices = np.argsort(scores)[::-1]

        # Build set of existing points for fast lookup
        existing_set = set()
        if X_existing is not None and X_existing.size > 0:
            for row in X_existing:
                existing_set.add(self._row_to_tuple(row))

        # Select top-k unique candidates
        selected = []
        selected_scores = []
        selected_set = set()

        for idx in sorted_indices:
            if len(selected) >= k:
                break

            candidate = X_candidate[idx]
            candidate_tuple = self._row_to_tuple(candidate)

            if candidate_tuple not in existing_set and candidate_tuple not in selected_set:
                selected.append(candidate)
                selected_scores.append(scores[idx])
                selected_set.add(candidate_tuple)

        if len(selected) == 0:
            return np.array([]).reshape(0, X_candidate.shape[1]), np.array([])

        return np.array(selected), np.array(selected_scores)

    def _row_to_tuple(self, row: np.ndarray) -> Tuple:
        """Convert a row to a tuple for hashing."""
        return tuple(np.round(row / self.tolerance) * self.tolerance)


# =============================================================================
# 6. ExperimentInterface: Abstract Interface for Running Experiments
# =============================================================================

class ExperimentInterface(ABC):
    """Abstract interface for executing real experiments."""

    @abstractmethod
    def run_experiments(self, X_next: np.ndarray) -> np.ndarray:
        """
        Execute experiments for given feature configurations.

        Args:
            X_next: Feature configurations, shape (n_experiments, n_features)

        Returns:
            Observed target values, shape (n_experiments,)
        """
        pass


class SimulatedExperimentInterface(ExperimentInterface):
    """
    Simulated experiment interface using a ground truth function.
    """

    def __init__(self,
                 objective_function: Callable[[np.ndarray], np.ndarray],
                 noise_std: float = 0.0):
        """
        Initialize with a ground truth objective function.

        Args:
            objective_function: Function that takes X and returns true y values
            noise_std: Standard deviation of Gaussian noise to add
        """
        self.objective_function = objective_function
        self.noise_std = noise_std

    def run_experiments(self, X_next: np.ndarray) -> np.ndarray:
        """Compute true values from the objective function."""
        if len(X_next.shape) == 1:
            X_next = X_next.reshape(1, -1)

        y_true = self.objective_function(X_next)

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=len(y_true))
            y_true = y_true + noise

        return y_true


class ManualExperimentInterface(ExperimentInterface):
    """
    Interface for manual experiment input.
    Results are set externally via set_results method.
    """

    def __init__(self):
        self._pending_X: Optional[np.ndarray] = None
        self._results: Optional[np.ndarray] = None

    def set_pending(self, X_next: np.ndarray) -> None:
        """Set pending experiments waiting for results."""
        self._pending_X = X_next
        self._results = None

    def set_results(self, y_next: np.ndarray) -> None:
        """Set results for pending experiments."""
        self._results = np.array(y_next, dtype=np.float64).flatten()

    def has_results(self) -> bool:
        """Check if results are available."""
        return self._results is not None

    def run_experiments(self, X_next: np.ndarray) -> np.ndarray:
        """Return the manually set results."""
        if self._results is None:
            raise RuntimeError("No results set. Call set_results first.")
        return self._results


# =============================================================================
# 7. ActiveLoop: Main Active Learning Controller
# =============================================================================

class ActiveLoop:
    """
    Orchestrates the complete active learning iteration cycle.
    """

    def __init__(self,
                 data_store: DataStore,
                 design_space: DesignSpace,
                 ensemble_model: EnsembleModel,
                 candidate_evaluator: CandidateEvaluator,
                 selector: Selector,
                 experiment_interface: ExperimentInterface,
                 n_candidates: int = 1000,
                 batch_size: int = 3):
        """
        Initialize the active learning loop.

        Args:
            data_store: Storage for labeled experiment data
            design_space: Defines the feature space and generates candidates
            ensemble_model: Ensemble for predictions and uncertainty
            candidate_evaluator: Acquisition function for scoring
            selector: Selects top candidates
            experiment_interface: Interface for running real experiments
            n_candidates: Number of candidates to generate each round
            batch_size: Number of experiments to run per round
        """
        self.data_store = data_store
        self.design_space = design_space
        self.ensemble_model = ensemble_model
        self.candidate_evaluator = candidate_evaluator
        self.selector = selector
        self.experiment_interface = experiment_interface

        self.n_candidates = n_candidates
        self.batch_size = batch_size

        # Tracking variables
        self.best_x: Optional[np.ndarray] = None
        self.best_y: float = -np.inf if candidate_evaluator.maximize else np.inf
        self.round_count: int = 0
        self.history: List[Dict[str, Any]] = []

        # Update best from existing data
        self._update_best()

    def _update_best(self) -> None:
        """Update the best observed point from current data."""
        X, y = self.data_store.get_labeled_data()
        if len(y) > 0:
            if self.candidate_evaluator.maximize:
                best_idx = np.argmax(y)
                if y[best_idx] > self.best_y:
                    self.best_y = y[best_idx]
                    self.best_x = X[best_idx].copy()
            else:
                best_idx = np.argmin(y)
                if y[best_idx] < self.best_y:
                    self.best_y = y[best_idx]
                    self.best_x = X[best_idx].copy()

            # Update evaluator's best_y for EI/PI
            self.candidate_evaluator.set_best_y(self.best_y)

    def suggest_next(self, n_suggestions: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Suggest next experiments without executing them.

        Returns:
            Tuple of (X_next, scores, mean_predictions, std_predictions)
        """
        if n_suggestions is None:
            n_suggestions = self.batch_size

        # Get current data
        X_labeled, y_labeled = self.data_store.get_labeled_data()

        # Train ensemble
        if len(y_labeled) >= 2:
            self.ensemble_model.fit_ensemble(X_labeled, y_labeled)

        # Generate candidates
        X_candidate = self.design_space.generate_candidates(
            n_candidates=self.n_candidates, mode="random"
        )

        # Get predictions
        if len(y_labeled) >= 2:
            mean, std = self.ensemble_model.predict_stats(X_candidate)
        else:
            # Not enough data, use random selection
            mean = np.zeros(len(X_candidate))
            std = np.ones(len(X_candidate))

        # Score candidates
        scores = self.candidate_evaluator.evaluate(mean, std)

        # Select top candidates
        X_next, selected_scores = self.selector.select(
            X_candidate, scores, k=n_suggestions, X_existing=X_labeled
        )

        # Get predictions for selected
        if len(X_next) > 0 and len(y_labeled) >= 2:
            mean_next, std_next = self.ensemble_model.predict_stats(X_next)
        else:
            mean_next = np.zeros(len(X_next)) if len(X_next) > 0 else np.array([])
            std_next = np.zeros(len(X_next)) if len(X_next) > 0 else np.array([])

        return X_next, selected_scores, mean_next, std_next

    def add_results(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """
        Add new experiment results and update state.

        Args:
            X_new: Feature configurations, shape (n_new, n_features)
            y_new: Observed target values, shape (n_new,)

        Returns:
            Dictionary with round information
        """
        self.round_count += 1

        # Add to data store
        self.data_store.add_observations(X_new, y_new)

        # Update best
        old_best_y = self.best_y
        self._update_best()

        # Record history
        round_info = {
            "round": self.round_count,
            "n_added": len(y_new),
            "X_new": X_new.copy() if isinstance(X_new, np.ndarray) else np.array(X_new),
            "y_new": y_new.copy() if isinstance(y_new, np.ndarray) else np.array(y_new),
            "best_x": self.best_x.copy() if self.best_x is not None else None,
            "best_y": self.best_y,
            "improved": self.best_y != old_best_y,
            "dataset_size": self.data_store.n_samples
        }
        self.history.append(round_info)

        return round_info

    def get_best(self) -> Tuple[Optional[np.ndarray], float]:
        """Get the current best observed configuration and value."""
        return self.best_x, self.best_y

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the complete history of all rounds."""
        return self.history

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all labeled data."""
        return self.data_store.get_labeled_data()

    def reset(self) -> None:
        """Reset the active learning loop."""
        self.data_store.reset()
        self.best_x = None
        self.best_y = -np.inf if self.candidate_evaluator.maximize else np.inf
        self.round_count = 0
        self.history.clear()
