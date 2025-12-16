"""
Active Learning Module for Experimental Design / Feature Combination Search

This module implements an active learning pipeline that combines ensemble models
with acquisition functions to iteratively find optimal feature combinations.

Key Components:
- DataStore: Manages labeled experiment data
- DesignSpace: Defines and generates candidate feature combinations
- BaseModelWrapper / EnsembleModel: Model abstraction and ensemble predictions
- CandidateEvaluator: Scores candidates using various acquisition strategies
- Selector: Selects top candidates while avoiding duplicates
- ExperimentInterface: Abstract interface for running real experiments
- ActiveLoop: Orchestrates the complete active learning iteration

Author: AI Assistant
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union
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
    """

    def __init__(self, X_init: Optional[np.ndarray] = None,
                 y_init: Optional[np.ndarray] = None):
        """
        Initialize the DataStore with optional initial data.

        Args:
            X_init: Initial feature matrix, shape (n_samples, n_features)
            y_init: Initial target values, shape (n_samples,)
        """
        if X_init is not None and y_init is not None:
            self.X_labeled = np.array(X_init, dtype=np.float64)
            self.y_labeled = np.array(y_init, dtype=np.float64).flatten()
        else:
            self.X_labeled = np.array([]).reshape(0, 0)
            self.y_labeled = np.array([])

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


# =============================================================================
# 2. DesignSpace: Feature Design Space and Candidate Generation
# =============================================================================

class FeatureDimension:
    """
    Represents a single feature dimension in the design space.

    Supports both continuous (bounded interval) and discrete (enumerated values)
    feature types.
    """

    def __init__(self, name: str, bounds: Union[Tuple[float, float, str], List]):
        """
        Initialize a feature dimension.

        Args:
            name: Name of the feature
            bounds: Either (min, max, "continuous") for continuous features,
                   or a list of discrete values [v1, v2, ...]
        """
        self.name = name

        if isinstance(bounds, tuple) and len(bounds) == 3 and bounds[2] == "continuous":
            self.type = "continuous"
            self.min_val = float(bounds[0])
            self.max_val = float(bounds[1])
            self.values = None
        elif isinstance(bounds, (list, np.ndarray)):
            self.type = "discrete"
            self.values = list(bounds)
            self.min_val = min(self.values)
            self.max_val = max(self.values)
        else:
            raise ValueError(f"Invalid bounds specification for feature '{name}': {bounds}")

    def sample_random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate random samples from this dimension.

        Args:
            n: Number of samples
            rng: Random number generator

        Returns:
            Array of sampled values, shape (n,)
        """
        if self.type == "continuous":
            return rng.uniform(self.min_val, self.max_val, size=n)
        else:
            return rng.choice(self.values, size=n)

    def get_grid(self, n_points: int) -> np.ndarray:
        """
        Get grid points for this dimension.

        Args:
            n_points: Number of points for continuous features (ignored for discrete)

        Returns:
            Array of grid values
        """
        if self.type == "continuous":
            return np.linspace(self.min_val, self.max_val, n_points)
        else:
            return np.array(self.values)


class DesignSpace:
    """
    Defines the feature design space and generates candidate combinations.

    Manages multiple feature dimensions and provides methods to generate
    candidate points through random sampling or grid-based approaches.
    """

    def __init__(self, dimensions: List[Tuple[str, Union[Tuple, List]]]):
        """
        Initialize the design space.

        Args:
            dimensions: List of (name, bounds) tuples defining each feature.
                       For continuous: ("feature_name", (min, max, "continuous"))
                       For discrete: ("feature_name", [v1, v2, v3, ...])

        Example:
            DesignSpace([
                ("x1", (-5, 5, "continuous")),
                ("x2", (-5, 5, "continuous")),
                ("material", ["A", "B", "C"])
            ])
        """
        self.dimensions = [FeatureDimension(name, bounds) for name, bounds in dimensions]
        self.n_features = len(self.dimensions)
        self.feature_names = [d.name for d in self.dimensions]

    def generate_candidates(self, n_candidates: int,
                           mode: str = "random",
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Generate candidate feature combinations.

        Args:
            n_candidates: Number of candidates to generate
            mode: Generation mode - "random" for random sampling,
                 "grid" for grid-based (number per dim = n_candidates^(1/n_dims))
            seed: Random seed for reproducibility

        Returns:
            Candidate matrix of shape (n_candidates, n_features)
        """
        rng = np.random.default_rng(seed)

        if mode == "random":
            return self._generate_random(n_candidates, rng)
        elif mode == "grid":
            return self._generate_grid(n_candidates)
        else:
            raise ValueError(f"Unknown generation mode: {mode}. Use 'random' or 'grid'.")

    def _generate_random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n random candidates."""
        samples = np.zeros((n, self.n_features))
        for i, dim in enumerate(self.dimensions):
            samples[:, i] = dim.sample_random(n, rng)
        return samples

    def _generate_grid(self, n_total: int) -> np.ndarray:
        """Generate grid-based candidates."""
        # Compute points per dimension to approximate n_total
        n_per_dim = max(2, int(np.ceil(n_total ** (1.0 / self.n_features))))

        # Generate grids for each dimension
        grids = [dim.get_grid(n_per_dim) for dim in self.dimensions]

        # Create meshgrid and reshape
        meshes = np.meshgrid(*grids, indexing='ij')
        candidates = np.stack([m.flatten() for m in meshes], axis=1)

        # If too many, subsample
        if len(candidates) > n_total:
            indices = np.random.choice(len(candidates), size=n_total, replace=False)
            candidates = candidates[indices]

        return candidates

    def is_valid(self, X: np.ndarray) -> np.ndarray:
        """
        Check which candidates are within the valid design space.

        Args:
            X: Candidate matrix of shape (n, n_features)

        Returns:
            Boolean array of shape (n,) indicating validity
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        valid = np.ones(len(X), dtype=bool)
        for i, dim in enumerate(self.dimensions):
            if dim.type == "continuous":
                valid &= (X[:, i] >= dim.min_val) & (X[:, i] <= dim.max_val)
            else:
                valid &= np.isin(X[:, i], dim.values)

        return valid


# =============================================================================
# 3. Model Wrappers: BaseModelWrapper and EnsembleModel
# =============================================================================

class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.

    Provides a unified interface for different regression models,
    allowing the active learning logic to be decoupled from specific
    model implementations.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on given data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        pass


class SklearnRegressorWrapper(BaseModelWrapper):
    """
    Wrapper for scikit-learn regression models.

    Allows any sklearn regressor to be used within the active learning
    framework through a consistent interface.
    """

    def __init__(self, model):
        """
        Initialize the wrapper with a sklearn model.

        Args:
            model: A scikit-learn regressor instance (e.g., RandomForestRegressor)
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

    Creates multiple base models, each trained on a bootstrap sample of
    the data, to enable uncertainty estimation through prediction variance.

    Attributes:
        models (List[BaseModelWrapper]): List of trained base models
        n_estimators (int): Number of models in the ensemble
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

        Each model is trained on a bootstrap sample (with replacement)
        of the original data.

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

        Args:
            X_candidate: Candidate matrix, shape (n_candidates, n_features)

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

        Args:
            X_candidate: Candidate matrix, shape (n_candidates, n_features)

        Returns:
            Tuple of (mean, std) where each has shape (n_candidates,)
        """
        predictions = self.predict_all(X_candidate)  # (n_estimators, n_candidates)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std


# =============================================================================
# 4. CandidateEvaluator: Acquisition Function / Scoring Strategy
# =============================================================================

class CandidateEvaluator:
    """
    Evaluates candidates using various acquisition strategies.

    Computes a score for each candidate point based on the predicted
    mean and uncertainty, balancing exploration and exploitation.

    Supported modes:
        - "mean": Pure exploitation (score = mean)
        - "std": Pure exploration (score = std)
        - "ucb": Upper Confidence Bound (score = mean + lambda * std)
    """

    def __init__(self, mode: str = "ucb", ucb_lambda: float = 1.0):
        """
        Initialize the evaluator.

        Args:
            mode: Evaluation strategy - "mean", "std", or "ucb"
            ucb_lambda: Weight for uncertainty in UCB mode (λ in mean + λ*std)
        """
        valid_modes = ["mean", "std", "ucb"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}")

        self.mode = mode
        self.ucb_lambda = ucb_lambda

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
            # Pure exploitation: prefer high predicted values
            return mean
        elif self.mode == "std":
            # Pure exploration: prefer high uncertainty
            return std
        elif self.mode == "ucb":
            # UCB: balance exploitation and exploration
            return mean + self.ucb_lambda * std
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# 5. Selector: Select Top Candidates for Next Experiments
# =============================================================================

class Selector:
    """
    Selects top-k candidates based on scores while avoiding duplicates.

    Ensures that selected candidates are not already in the existing
    dataset to avoid redundant experiments.
    """

    def __init__(self, tolerance: float = 1e-8):
        """
        Initialize the selector.

        Args:
            tolerance: Tolerance for floating-point comparison when checking
                      for duplicate feature vectors
        """
        self.tolerance = tolerance

    def select(self,
               X_candidate: np.ndarray,
               scores: np.ndarray,
               k: int,
               X_existing: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select top-k candidates by score, excluding existing points.

        Args:
            X_candidate: Candidate matrix, shape (n_candidates, n_features)
            scores: Score for each candidate, shape (n_candidates,)
            k: Number of candidates to select
            X_existing: Existing feature matrix to exclude, shape (n_existing, n_features)

        Returns:
            Selected candidates, shape (k, n_features) or fewer if not enough unique
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
        selected_set = set()

        for idx in sorted_indices:
            if len(selected) >= k:
                break

            candidate = X_candidate[idx]
            candidate_tuple = self._row_to_tuple(candidate)

            # Check if not in existing data and not already selected
            if candidate_tuple not in existing_set and candidate_tuple not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate_tuple)

        if len(selected) == 0:
            return np.array([]).reshape(0, X_candidate.shape[1])

        return np.array(selected)

    def _row_to_tuple(self, row: np.ndarray) -> Tuple:
        """Convert a row to a tuple for hashing, with rounding for float tolerance."""
        return tuple(np.round(row / self.tolerance) * self.tolerance)


# =============================================================================
# 6. ExperimentInterface: Abstract Interface for Running Experiments
# =============================================================================

class ExperimentInterface(ABC):
    """
    Abstract interface for executing real experiments.

    Provides a contract for how the active learning loop obtains
    true target values for selected feature combinations.
    """

    @abstractmethod
    def run_experiments(self, X_next: np.ndarray) -> np.ndarray:
        """
        Execute experiments for given feature configurations.

        Args:
            X_next: Feature configurations to evaluate, shape (n_experiments, n_features)

        Returns:
            Observed target values, shape (n_experiments,)
        """
        pass


class ConsoleExperimentInterface(ExperimentInterface):
    """
    Interactive console-based experiment interface.

    Prompts the user to manually enter experimental results
    for each candidate point via command line input.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the console interface.

        Args:
            feature_names: Optional list of feature names for display
        """
        self.feature_names = feature_names

    def run_experiments(self, X_next: np.ndarray) -> np.ndarray:
        """
        Prompt user to enter experimental results via console.

        Args:
            X_next: Feature configurations, shape (n_experiments, n_features)

        Returns:
            User-entered target values, shape (n_experiments,)
        """
        if len(X_next.shape) == 1:
            X_next = X_next.reshape(1, -1)

        n_experiments = len(X_next)
        results = np.zeros(n_experiments)

        print("\n" + "=" * 60)
        print("EXPERIMENT INPUT REQUIRED")
        print("=" * 60)

        for i, x in enumerate(X_next):
            print(f"\nExperiment {i + 1}/{n_experiments}:")
            if self.feature_names:
                for name, val in zip(self.feature_names, x):
                    print(f"  {name}: {val:.4f}")
            else:
                print(f"  Features: {x}")

            while True:
                try:
                    y_str = input("  Enter observed result (y): ")
                    results[i] = float(y_str)
                    break
                except ValueError:
                    print("  Invalid input. Please enter a numeric value.")

        print("=" * 60 + "\n")
        return results


class SimulatedExperimentInterface(ExperimentInterface):
    """
    Simulated experiment interface using a ground truth function.

    Useful for testing and demonstration purposes where a known
    objective function can be used instead of real experiments.
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
        """
        Compute true values from the objective function.

        Args:
            X_next: Feature configurations, shape (n_experiments, n_features)

        Returns:
            True target values (with optional noise), shape (n_experiments,)
        """
        if len(X_next.shape) == 1:
            X_next = X_next.reshape(1, -1)

        y_true = self.objective_function(X_next)

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=len(y_true))
            y_true = y_true + noise

        return y_true


# =============================================================================
# 7. ActiveLoop: Main Active Learning Controller
# =============================================================================

class ActiveLoop:
    """
    Orchestrates the complete active learning iteration cycle.

    Coordinates all components to iteratively:
    1. Train ensemble on current data
    2. Generate and score candidates
    3. Select most promising points
    4. Execute experiments
    5. Update dataset and track best results

    Attributes:
        best_x (np.ndarray): Current best feature combination
        best_y (float): Current best observed value
        history (List[dict]): Record of each iteration's results
    """

    def __init__(self,
                 data_store: DataStore,
                 design_space: DesignSpace,
                 ensemble_model: EnsembleModel,
                 candidate_evaluator: CandidateEvaluator,
                 selector: Selector,
                 experiment_interface: ExperimentInterface,
                 n_candidates: int = 1000,
                 batch_size: int = 3,
                 verbose: bool = True):
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
            verbose: Whether to print progress information
        """
        self.data_store = data_store
        self.design_space = design_space
        self.ensemble_model = ensemble_model
        self.candidate_evaluator = candidate_evaluator
        self.selector = selector
        self.experiment_interface = experiment_interface

        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.verbose = verbose

        # Initialize tracking variables
        self.best_x: Optional[np.ndarray] = None
        self.best_y: float = -np.inf
        self.round_count: int = 0
        self.history: List[dict] = []

        # Initialize best from existing data
        self._update_best()

    def _update_best(self) -> None:
        """Update the best observed point from current data."""
        X, y = self.data_store.get_labeled_data()
        if len(y) > 0:
            best_idx = np.argmax(y)
            if y[best_idx] > self.best_y:
                self.best_y = y[best_idx]
                self.best_x = X[best_idx].copy()

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def run_one_round(self) -> dict:
        """
        Execute one complete round of active learning.

        Returns:
            Dictionary containing round statistics and results
        """
        self.round_count += 1
        self._log(f"\n{'=' * 60}")
        self._log(f"ACTIVE LEARNING ROUND {self.round_count}")
        self._log(f"{'=' * 60}")

        # Step 1: Get current labeled data
        X_labeled, y_labeled = self.data_store.get_labeled_data()
        self._log(f"Current dataset size: {len(y_labeled)} samples")

        # Step 2: Train ensemble model
        self._log("Training ensemble model...")
        self.ensemble_model.fit_ensemble(X_labeled, y_labeled)

        # Step 3: Generate candidates
        self._log(f"Generating {self.n_candidates} candidates...")
        X_candidate = self.design_space.generate_candidates(
            n_candidates=self.n_candidates, mode="random"
        )

        # Step 4: Get predictions and uncertainty
        mean, std = self.ensemble_model.predict_stats(X_candidate)

        # Step 5: Score candidates
        scores = self.candidate_evaluator.evaluate(mean, std)

        # Step 6: Select next experiments
        X_next = self.selector.select(
            X_candidate, scores, k=self.batch_size, X_existing=X_labeled
        )

        if len(X_next) == 0:
            self._log("Warning: No new candidates selected (all may be duplicates)")
            return {"round": self.round_count, "n_selected": 0}

        self._log(f"\nSelected {len(X_next)} candidates for experiments:")

        # Get predictions for selected candidates for logging
        mean_next, std_next = self.ensemble_model.predict_stats(X_next)

        for i, (x, m, s) in enumerate(zip(X_next, mean_next, std_next)):
            self._log(f"  Candidate {i + 1}: {x} | pred_mean={m:.4f}, pred_std={s:.4f}")

        # Step 7: Run experiments
        self._log("\nRunning experiments...")
        y_next = self.experiment_interface.run_experiments(X_next)

        self._log("Experiment results:")
        for i, (x, y) in enumerate(zip(X_next, y_next)):
            self._log(f"  Result {i + 1}: {x} -> y = {y:.4f}")

        # Step 8: Update data store
        self.data_store.add_observations(X_next, y_next)

        # Step 9: Update best
        self._update_best()

        # Record history
        round_info = {
            "round": self.round_count,
            "n_selected": len(X_next),
            "X_next": X_next.copy(),
            "y_next": y_next.copy(),
            "best_x": self.best_x.copy() if self.best_x is not None else None,
            "best_y": self.best_y,
            "dataset_size": self.data_store.n_samples
        }
        self.history.append(round_info)

        self._log(f"\nCurrent best: y = {self.best_y:.4f} at x = {self.best_x}")

        return round_info

    def run(self, max_rounds: int) -> None:
        """
        Run multiple rounds of active learning.

        Args:
            max_rounds: Maximum number of rounds to execute
        """
        self._log(f"\nStarting Active Learning for {max_rounds} rounds")
        self._log(f"Acquisition strategy: {self.candidate_evaluator.mode}")

        for _ in range(max_rounds):
            self.run_one_round()

        self._log(f"\n{'=' * 60}")
        self._log("ACTIVE LEARNING COMPLETE")
        self._log(f"{'=' * 60}")
        self._log(f"Total rounds: {self.round_count}")
        self._log(f"Final dataset size: {self.data_store.n_samples}")
        self._log(f"Best result: y = {self.best_y:.4f}")
        self._log(f"Best configuration: {self.best_x}")

    def get_best(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the current best observed configuration and value.

        Returns:
            Tuple of (best_x, best_y) where best_x is the feature configuration
            and best_y is the corresponding observed value
        """
        return self.best_x, self.best_y

    def get_history(self) -> List[dict]:
        """
        Get the complete history of all rounds.

        Returns:
            List of dictionaries containing information about each round
        """
        return self.history


# =============================================================================
# 8. Convenience Factory Function
# =============================================================================

def create_active_learning_pipeline(
    X_init: np.ndarray,
    y_init: np.ndarray,
    dimensions: List[Tuple[str, Union[Tuple, List]]],
    experiment_interface: ExperimentInterface,
    n_estimators: int = 10,
    acquisition_mode: str = "ucb",
    ucb_lambda: float = 1.0,
    n_candidates: int = 1000,
    batch_size: int = 3,
    base_model_params: Optional[dict] = None,
    verbose: bool = True
) -> ActiveLoop:
    """
    Convenience factory to create a complete active learning pipeline.

    Args:
        X_init: Initial feature data, shape (n_init, n_features)
        y_init: Initial target values, shape (n_init,)
        dimensions: Feature space definition for DesignSpace
        experiment_interface: Interface for running experiments
        n_estimators: Number of models in ensemble
        acquisition_mode: "mean", "std", or "ucb"
        ucb_lambda: Lambda parameter for UCB acquisition
        n_candidates: Number of candidates per round
        batch_size: Number of experiments per round
        base_model_params: Parameters for RandomForestRegressor
        verbose: Whether to print progress

    Returns:
        Configured ActiveLoop instance ready to run
    """
    from sklearn.ensemble import RandomForestRegressor

    # Default model parameters
    if base_model_params is None:
        base_model_params = {"n_estimators": 50, "random_state": None}

    # Create components
    data_store = DataStore(X_init, y_init)
    design_space = DesignSpace(dimensions)

    def model_factory():
        return SklearnRegressorWrapper(RandomForestRegressor(**base_model_params))

    ensemble_model = EnsembleModel(model_factory, n_estimators=n_estimators)
    candidate_evaluator = CandidateEvaluator(mode=acquisition_mode, ucb_lambda=ucb_lambda)
    selector = Selector()

    # Create and return the active loop
    return ActiveLoop(
        data_store=data_store,
        design_space=design_space,
        ensemble_model=ensemble_model,
        candidate_evaluator=candidate_evaluator,
        selector=selector,
        experiment_interface=experiment_interface,
        n_candidates=n_candidates,
        batch_size=batch_size,
        verbose=verbose
    )


# =============================================================================
# 9. Example / Demo
# =============================================================================

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    print("=" * 70)
    print("ACTIVE LEARNING MODULE DEMONSTRATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Define a virtual ground truth function (for demonstration)
    # True optimum: x1=2, x2=-1, y_max=5 (without noise)
    # -------------------------------------------------------------------------
    def ground_truth_function(X: np.ndarray) -> np.ndarray:
        """
        Virtual objective function: y = -(x1 - 2)^2 - (x2 + 1)^2 + 5
        Optimal point: x1=2, x2=-1 with y=5
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        x1 = X[:, 0]
        x2 = X[:, 1]
        return -((x1 - 2) ** 2) - ((x2 + 1) ** 2) + 5

    # -------------------------------------------------------------------------
    # Generate initial random experiments
    # -------------------------------------------------------------------------
    np.random.seed(42)
    n_init = 5
    X_init = np.random.uniform(-5, 5, size=(n_init, 2))
    y_init = ground_truth_function(X_init) + np.random.normal(0, 0.1, n_init)

    print(f"\nInitial data ({n_init} points):")
    for i, (x, y) in enumerate(zip(X_init, y_init)):
        print(f"  Point {i + 1}: x1={x[0]:.3f}, x2={x[1]:.3f} -> y={y:.3f}")

    # -------------------------------------------------------------------------
    # Set up components
    # -------------------------------------------------------------------------

    # 1. DataStore
    data_store = DataStore(X_init, y_init)

    # 2. DesignSpace: x1 and x2 both in [-5, 5]
    design_space = DesignSpace([
        ("x1", (-5, 5, "continuous")),
        ("x2", (-5, 5, "continuous"))
    ])

    # 3. Model factory for ensemble
    def create_model():
        return SklearnRegressorWrapper(
            RandomForestRegressor(n_estimators=50, random_state=None)
        )

    # 4. EnsembleModel
    ensemble_model = EnsembleModel(
        base_model_factory=create_model,
        n_estimators=10
    )

    # 5. CandidateEvaluator with UCB strategy
    candidate_evaluator = CandidateEvaluator(mode="ucb", ucb_lambda=1.5)

    # 6. Selector
    selector = Selector()

    # 7. Simulated experiment interface (uses ground truth + noise)
    experiment_interface = SimulatedExperimentInterface(
        objective_function=ground_truth_function,
        noise_std=0.1
    )

    # 8. ActiveLoop
    active_loop = ActiveLoop(
        data_store=data_store,
        design_space=design_space,
        ensemble_model=ensemble_model,
        candidate_evaluator=candidate_evaluator,
        selector=selector,
        experiment_interface=experiment_interface,
        n_candidates=500,
        batch_size=3,
        verbose=True
    )

    # -------------------------------------------------------------------------
    # Run active learning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STARTING ACTIVE LEARNING OPTIMIZATION")
    print("True optimum: x1=2.0, x2=-1.0 with y=5.0")
    print("=" * 70)

    active_loop.run(max_rounds=5)

    # -------------------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------------------
    best_x, best_y = active_loop.get_best()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best found: x1={best_x[0]:.4f}, x2={best_x[1]:.4f}")
    print(f"Best observed y: {best_y:.4f}")
    print(f"True optimum: x1=2.0, x2=-1.0, y=5.0")
    print(f"Distance from optimum: {np.sqrt((best_x[0]-2)**2 + (best_x[1]+1)**2):.4f}")

    # Show convergence
    print("\nConvergence history (best_y per round):")
    for h in active_loop.get_history():
        print(f"  Round {h['round']}: best_y = {h['best_y']:.4f}")
