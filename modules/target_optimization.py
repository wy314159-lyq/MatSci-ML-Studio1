"""
Enhanced Target Optimization Module for MatSci-ML Studio
Provides various optimization algorithms for material property prediction
Features: Real-time visualization, Genetic Algorithm, Particle Swarm Optimization
Performance optimized with parallel processing and advanced visualization
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import itertools
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import traceback
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
                           QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
                           QTextEdit, QProgressBar, QSplitter, QGroupBox, QTabWidget,
                           QScrollArea, QFrame, QMessageBox, QFileDialog,
                           QFormLayout, QSlider, QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QBrush, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Enhanced matplotlib settings for better performance and visualization
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['animation.html'] = 'html5'

# Optional imports for optimization algorithms
try:
    from scipy.optimize import minimize, differential_evolution, basinhopping, NonlinearConstraint, LinearConstraint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# 在文件开头导入垃圾回收模块
import gc

# 可选导入psutil用于内存监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed, system monitoring will be disabled")

@dataclass
class OptimizationConstraint:
    """Enhanced constraint definition for optimization"""
    feature_indices: List[int]
    coefficients: List[float]
    operator: str  # '<=', '>=', '=='
    bound: float
    description: str
    constraint_type: str = "linear"  # "linear" or "nonlinear"

@dataclass
class FeatureBounds:
    """Enhanced feature bounds definition with categorical support"""
    min_value: float
    max_value: float
    is_fixed: bool = False
    fixed_value: Optional[float] = None
    feature_type: str = "continuous"  # "continuous", "integer", "categorical", "binary"
    categorical_values: Optional[List[Union[int, float]]] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization"""
        if self.feature_type == "categorical" and self.categorical_values is None:
            self.categorical_values = [0, 1]  # Default binary
        elif self.feature_type == "binary":
            self.categorical_values = [0, 1]
            self.min_value = 0
            self.max_value = 1

class GeneticAlgorithm:
    """Enhanced Genetic Algorithm with real-time progress tracking"""
    
    def __init__(self, objective_func, bounds, population_size=50, max_generations=100, 
                 mutation_rate=0.1, crossover_rate=0.8, elite_size=5, callback=None, feature_types=None):
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.callback = callback
        self.dimension = len(bounds)
        self.should_stop = False
        
        # **HYBRID GA ENHANCEMENT**: Store feature types for mixed variable handling
        # Default to 'continuous' for backward compatibility
        if feature_types is None:
            self.feature_types = ['continuous'] * self.dimension
        else:
            self.feature_types = feature_types
            
        # Validate feature types length matches bounds
        if len(self.feature_types) != self.dimension:
            print(f"[WARNING] Feature types length ({len(self.feature_types)}) doesn't match bounds ({self.dimension}). Using continuous for all.")
            self.feature_types = ['continuous'] * self.dimension
            
        # Count different feature types for debugging
        continuous_count = sum(1 for ft in self.feature_types if ft == 'continuous')
        categorical_count = sum(1 for ft in self.feature_types if ft in ['categorical', 'binary', 'integer'])
        print(f"[HYBRID GA] Initialized with {continuous_count} continuous and {categorical_count} discrete features")
        
        # Real-time tracking
        self.generation_history = []
        self.best_fitness_history = []
        self.population_diversity = []
        
    def initialize_population(self):
        """Initialize random population within bounds with hybrid feature type support"""
        print(f"[GA INIT] Initializing population with {len(self.bounds)} bounds")
        print(f"[GA INIT] First 5 bounds: {self.bounds[:5]}")
        print(f"[GA INIT] Feature types: {self.feature_types[:5]}...")
        
        population = []
        for _ in range(self.population_size):
            individual = []
            for i, (min_val, max_val) in enumerate(self.bounds):
                if min_val == max_val:
                    # Fixed parameter
                    individual.append(min_val)
                elif self.feature_types[i] == 'continuous':
                    # Continuous feature: use uniform distribution
                    value = np.random.uniform(min_val, max_val)
                    individual.append(value)
                else:
                    # Categorical/binary/integer feature: use discrete uniform selection
                    # Generate random integer within bounds (inclusive)
                    discrete_value = np.random.randint(int(min_val), int(max_val) + 1)
                    individual.append(float(discrete_value))  # Convert to float for consistency
            population.append(np.array(individual))
            
        # Debug first individual
        if population:
            print(f"[GA INIT] First individual (first 5 features): {population[0][:5]}")
            
        return population
    
    def evaluate_population(self, population):
        """Evaluate fitness of entire population with thread-safe processing"""
        try:
            # **CRITICAL FIX**: Disable parallel processing to prevent SHAP conflicts
            # Always use sequential processing to avoid thread conflicts with SHAP
            fitness_values = []
            for individual in population:
                try:
                    fitness_value = self.objective_func(individual)
                    fitness_values.append(fitness_value)
                except Exception:
                    fitness_values.append(float('inf'))  # Fallback for failed evaluations
            
            return np.array(fitness_values)
        except Exception:
            # Ultimate fallback - return infinite fitness for all
            return np.full(len(population), float('inf'))
    
    def selection(self, population, fitness_values):
        """Tournament selection with diversity preservation"""
        selected = []
        tournament_size = max(2, self.population_size // 10)
        
        for _ in range(self.population_size - self.elite_size):
            # Tournament selection
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Add elite individuals
        elite_indices = np.argsort(fitness_values)[:self.elite_size]
        for idx in elite_indices:
            selected.append(population[idx].copy())
            
        return selected
    
    def crossover(self, parent1, parent2):
        """Hybrid crossover: SBX for continuous features, uniform crossover for discrete features"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        eta = 2.0  # Distribution index for SBX
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if self.feature_types[i] == 'continuous':
                # **CONTINUOUS FEATURE**: Use Simulated Binary Crossover (SBX)
                # This preserves the powerful continuous optimization capabilities
                if np.random.random() <= 0.5:
                    if abs(parent1[i] - parent2[i]) > 1e-14:
                        if parent1[i] < parent2[i]:
                            y1, y2 = parent1[i], parent2[i]
                        else:
                            y1, y2 = parent2[i], parent1[i]
                        
                        yl, yu = self.bounds[i][0], self.bounds[i][1]
                        
                        rand = np.random.random()
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                        alpha = 2.0 - beta ** -(eta + 1.0)
                        
                        if rand <= (1.0 / alpha):
                            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                        
                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                        
                        child1[i] = np.clip(c1, yl, yu)
                        child2[i] = np.clip(c2, yl, yu)
            else:
                # **DISCRETE FEATURE**: Use Uniform Crossover
                # Simple and effective for categorical/binary/integer features
                if np.random.random() <= 0.5:
                    # Swap the values between parents
                    child1[i], child2[i] = parent2[i], parent1[i]
                # If not swapping, children inherit parent values directly
                
                # Ensure discrete values remain integers
                yl, yu = self.bounds[i][0], self.bounds[i][1]
                child1[i] = float(int(np.clip(round(child1[i]), yl, yu)))
                child2[i] = float(int(np.clip(round(child2[i]), yl, yu)))
        
        return child1, child2
    
    def mutate(self, individual):
        """Hybrid mutation: polynomial for continuous features, random resetting for discrete features"""
        eta = 20.0  # Distribution index for polynomial mutation
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() <= self.mutation_rate:
                yl, yu = self.bounds[i][0], self.bounds[i][1]
                
                if yl == yu:  # Fixed parameter
                    continue
                
                if self.feature_types[i] == 'continuous':
                    # **CONTINUOUS FEATURE**: Use Polynomial Mutation
                    # This preserves the sophisticated continuous search capabilities
                    y = individual[i]
                    
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    
                    rand = np.random.random()
                    mut_pow = 1.0 / (eta + 1.0)
                    
                    if rand <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                        deltaq = 1.0 - val ** mut_pow
                    
                    y = y + deltaq * (yu - yl)
                    mutated[i] = np.clip(y, yl, yu)
                    
                else:
                    # **DISCRETE FEATURE**: Use Random Resetting Mutation
                    # Randomly select a new valid discrete value
                    current_value = int(round(individual[i]))
                    possible_values = list(range(int(yl), int(yu) + 1))
                    
                    # Remove current value to ensure actual mutation
                    if current_value in possible_values:
                        possible_values.remove(current_value)
                    
                    # If there are other valid values, randomly select one
                    if possible_values:
                        new_value = np.random.choice(possible_values)
                        mutated[i] = float(new_value)
                    # If no other values available (e.g., binary with only 2 options), 
                    # just flip to the other value
                    elif len(range(int(yl), int(yu) + 1)) == 2:
                        all_values = list(range(int(yl), int(yu) + 1))
                        other_value = [v for v in all_values if v != current_value][0]
                        mutated[i] = float(other_value)
        
        return mutated
    
    def calculate_diversity(self, population):
        """Calculate population diversity for monitoring convergence"""
        if len(population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(population[i] - population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def optimize(self):
        """Main optimization loop with enhanced real-time tracking"""
        # Initialize population
        population = self.initialize_population()
        
        best_individual = None
        best_fitness = float('inf')
        
        # Ensure we have at least initial population evaluation
        if population:
            fitness_values = self.evaluate_population(population)
            min_fitness_idx = np.argmin(fitness_values)
            best_fitness = fitness_values[min_fitness_idx]
            best_individual = population[min_fitness_idx].copy()
        
        for generation in range(self.max_generations):
            if self.should_stop:
                break
            
            # Evaluate population
            fitness_values = self.evaluate_population(population)
            
            # Update best
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < best_fitness:
                best_fitness = fitness_values[min_fitness_idx]
                best_individual = population[min_fitness_idx].copy()
            
            # Calculate diversity
            diversity = self.calculate_diversity(population)
            
            # Store history for visualization
            self.generation_history.append(generation)
            self.best_fitness_history.append(best_fitness)
            self.population_diversity.append(diversity)
            
            # Callback for real-time updates
            if self.callback:
                self.callback(generation, best_fitness, best_individual, {
                    'population_fitness': fitness_values,
                    'diversity': diversity,
                    'generation': generation
                })
            
            # Selection
            selected = self.selection(population, fitness_values)
            
            # Create next generation
            new_population = []
            
            # Keep elite
            elite_indices = np.argsort(fitness_values)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                if len(selected) >= 2:
                    parent1, parent2 = np.random.choice(len(selected), 2, replace=False)
                    child1, child2 = self.crossover(selected[parent1], selected[parent2])
                    
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.append(child1)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)
                else:
                    break
            
            population = new_population[:self.population_size]
        
        # Ensure we have a valid result even if optimization was stopped early
        if best_individual is None and population:
            # Use the first individual if no best was found
            best_individual = population[0].copy()
            best_fitness = self.objective_func(best_individual)
        
        # Final fallback - create a random individual within bounds
        if best_individual is None:
            best_individual = []
            for i, (min_val, max_val) in enumerate(self.bounds):
                if min_val == max_val:
                    best_individual.append(min_val)
                elif self.feature_types[i] == 'continuous':
                    best_individual.append(np.random.uniform(min_val, max_val))
                else:
                    # For categorical/integer features
                    discrete_value = np.random.randint(int(min_val), int(max_val) + 1)
                    best_individual.append(float(discrete_value))
            best_individual = np.array(best_individual)
            best_fitness = self.objective_func(best_individual)
        
        # **CRITICAL FIX**: Enforce bounds on final result to prevent boundary violations
        if best_individual is not None:
            for i in range(len(best_individual)):
                if i < len(self.bounds):
                    min_val, max_val = self.bounds[i]
                    if self.feature_types[i] == 'continuous':
                        # For continuous features: strict boundary clamping
                        best_individual[i] = np.clip(best_individual[i], min_val, max_val)
                    else:
                        # For discrete features: round and clamp
                        best_individual[i] = float(int(np.clip(round(best_individual[i]), min_val, max_val)))
            
            print(f"[GA FINAL] Final result bounds verification:")
            print(f"[GA FINAL] Solution: {best_individual[:5]}...")
            violations = []
            for i, (value, (min_val, max_val)) in enumerate(zip(best_individual, self.bounds)):
                if value < min_val or value > max_val:
                    violations.append(f"Feature {i}: {value:.6f} not in [{min_val:.6f}, {max_val:.6f}]")
            
            if violations:
                print(f"[GA FINAL] ERROR: Found {len(violations)} boundary violations:")
                for violation in violations[:3]:  # Show first 3
                    print(f"[GA FINAL]   {violation}")
            else:
                print(f"[GA FINAL] SUCCESS: All bounds satisfied")
        
        return {
            'x': best_individual,
            'fun': best_fitness,
            'nfev': self.max_generations * self.population_size,
            'nit': self.max_generations
        }

class ParticleSwarmOptimization:
    """Enhanced Particle Swarm Optimization with real-time progress tracking"""
    
    def __init__(self, objective_func, bounds, n_particles=30, max_iterations=100,
                 w=0.9, c1=2.0, c2=2.0, callback=None, feature_types=None):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.callback = callback
        self.dimension = len(bounds)
        self.should_stop = False
        
        # **HYBRID PSO ENHANCEMENT**: Store feature types for mixed variable handling
        if feature_types is None:
            self.feature_types = ['continuous'] * self.dimension
        else:
            self.feature_types = feature_types
            
        # Count different feature types for debugging
        continuous_count = sum(1 for ft in self.feature_types if ft == 'continuous')
        categorical_count = sum(1 for ft in self.feature_types if ft in ['categorical', 'binary', 'integer'])
        print(f"[HYBRID PSO] Initialized with {continuous_count} continuous and {categorical_count} discrete features")
        
        # Real-time tracking
        self.iteration_history = []
        self.best_fitness_history = []
        self.swarm_positions = []
        self.velocities_magnitude = []
        
    def initialize_swarm(self):
        """Initialize particle swarm with hybrid feature type support"""
        positions = []
        velocities = []
        
        for _ in range(self.n_particles):
            # Initialize position
            position = []
            velocity = []
            for i, (min_val, max_val) in enumerate(self.bounds):
                if min_val == max_val:
                    position.append(min_val)
                    velocity.append(0.0)
                elif self.feature_types[i] == 'continuous':
                    # Continuous feature: use uniform distribution
                    position.append(np.random.uniform(min_val, max_val))
                    velocity.append(np.random.uniform(-(max_val - min_val) * 0.1, 
                                                    (max_val - min_val) * 0.1))
                else:
                    # Discrete feature: use discrete uniform selection
                    discrete_value = np.random.randint(int(min_val), int(max_val) + 1)
                    position.append(float(discrete_value))
                    # For discrete features, use smaller velocity range
                    velocity.append(np.random.uniform(-0.5, 0.5))
            positions.append(np.array(position))
            velocities.append(np.array(velocity))
        
        return positions, velocities
    
    def evaluate_swarm(self, positions):
        """Evaluate fitness of entire swarm with thread-safe processing"""
        try:
            # **CRITICAL FIX**: Disable parallel processing to prevent SHAP conflicts
            # Always use sequential processing to avoid thread conflicts with SHAP
            fitness_values = []
            for position in positions:
                try:
                    fitness_value = self.objective_func(position)
                    fitness_values.append(fitness_value)
                except Exception:
                    fitness_values.append(float('inf'))  # Fallback for failed evaluations
            
            return np.array(fitness_values)
        except Exception:
            # Ultimate fallback - return infinite fitness for all
            return np.full(len(positions), float('inf'))
    
    def update_velocity(self, velocity, position, personal_best, global_best):
        """Update particle velocity with adaptive parameters"""
        r1, r2 = np.random.random(self.dimension), np.random.random(self.dimension)
        
        # Adaptive inertia weight (decreases over time)
        cognitive = self.c1 * r1 * (personal_best - position)
        social = self.c2 * r2 * (global_best - position)
        
        new_velocity = self.w * velocity + cognitive + social
        
        # Velocity clamping
        for i, (min_val, max_val) in enumerate(self.bounds):
            if min_val != max_val:
                v_max = (max_val - min_val) * 0.2
                new_velocity[i] = np.clip(new_velocity[i], -v_max, v_max)
        
        return new_velocity
    
    def update_position(self, position, velocity):
        """Update particle position with hybrid feature type support"""
        new_position = position + velocity
        
        # Boundary handling with feature type awareness
        for i, (min_val, max_val) in enumerate(self.bounds):
            if self.feature_types[i] == 'continuous':
                # Continuous feature: standard boundary clamping
                if new_position[i] < min_val:
                    new_position[i] = min_val
                elif new_position[i] > max_val:
                    new_position[i] = max_val
            else:
                # Discrete feature: round to nearest integer and clamp
                new_position[i] = float(int(np.clip(round(new_position[i]), min_val, max_val)))
        
        return new_position
    
    def calculate_swarm_diversity(self, positions):
        """Calculate swarm diversity for convergence monitoring"""
        if len(positions) < 2:
            return 0.0
        
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        return np.mean(distances)
    
    def optimize(self):
        """Main optimization loop with enhanced real-time tracking"""
        # Initialize swarm
        positions, velocities = self.initialize_swarm()
        
        if not positions:
            # Emergency fallback if no positions were created
            position = []
            for min_val, max_val in self.bounds:
                if min_val == max_val:
                    position.append(min_val)
                else:
                    position.append(np.random.uniform(min_val, max_val))
            return {
                'x': np.array(position),
                'fun': float('inf'),
                'nfev': 0,
                'nit': 0
            }
        
        # Initialize personal and global bests
        fitness_values = self.evaluate_swarm(positions)
        personal_bests = [pos.copy() for pos in positions]
        personal_best_fitness = fitness_values.copy()
        
        global_best_idx = np.argmin(fitness_values)
        global_best = positions[global_best_idx].copy()
        global_best_fitness = fitness_values[global_best_idx]
        
        for iteration in range(self.max_iterations):
            if self.should_stop:
                break
            
            # Update inertia weight (linear decrease)
            self.w = 0.9 - (0.9 - 0.4) * iteration / self.max_iterations
            
            # Update velocities and positions
            for i in range(self.n_particles):
                velocities[i] = self.update_velocity(
                    velocities[i], positions[i], personal_bests[i], global_best
                )
                positions[i] = self.update_position(positions[i], velocities[i])
            
            # Evaluate new positions
            fitness_values = self.evaluate_swarm(positions)
            
            # Update personal bests
            for i in range(self.n_particles):
                if fitness_values[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness_values[i]
                    personal_bests[i] = positions[i].copy()
            
            # Update global best
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < global_best_fitness:
                global_best_fitness = fitness_values[min_fitness_idx]
                global_best = positions[min_fitness_idx].copy()
            
            # Calculate metrics for visualization
            diversity = self.calculate_swarm_diversity(positions)
            avg_velocity = np.mean([np.linalg.norm(v) for v in velocities])
            
            # Store history
            self.iteration_history.append(iteration)
            self.best_fitness_history.append(global_best_fitness)
            self.swarm_positions.append([pos.copy() for pos in positions])
            self.velocities_magnitude.append(avg_velocity)
            
            # Callback for real-time updates
            if self.callback:
                self.callback(iteration, global_best_fitness, global_best, {
                    'positions': positions,
                    'velocities': velocities,
                    'diversity': diversity,
                    'avg_velocity': avg_velocity,
                    'iteration': iteration
                })
        
        # Ensure we have a valid result
        if global_best is None:
            # Use the first position if no global best was set
            global_best = positions[0].copy()
            global_best_fitness = fitness_values[0]
        
        return {
            'x': global_best,
            'fun': global_best_fitness,
            'nfev': self.max_iterations * self.n_particles,
            'nit': self.max_iterations
        }

class ThreadSafeModelWrapper:
    """Thread-safe wrapper for models to prevent conflicts with SHAP"""
    def __init__(self, model, feature_names=None):
        """Initialize thread-safe model wrapper with enhanced feature name handling"""
        self.model = model
        self._lock = threading.Lock()
        
        # **CRITICAL FIX**: Prioritize model's actual feature names
        model_feature_names = None
        
        # Try to extract feature names from the model itself first
        if hasattr(self.model, 'feature_names_in_'):
            model_feature_names = list(self.model.feature_names_in_)
            print(f"[DEBUG] ThreadSafeModelWrapper: Got feature names from model.feature_names_in_")
        elif hasattr(self.model, 'named_steps'):
            # For Pipeline, try to get from the first step (usually preprocessor)
            try:
                first_step_name, first_step = list(self.model.named_steps.items())[0]
                if hasattr(first_step, 'feature_names_in_'):
                    model_feature_names = list(first_step.feature_names_in_)
                    print(f"[DEBUG] ThreadSafeModelWrapper: Got feature names from {first_step_name}.feature_names_in_")
            except Exception as e:
                print(f"[DEBUG] ThreadSafeModelWrapper: Could not extract from Pipeline: {e}")
        
        # Use feature names in order of priority:
        # 1. Model's actual feature names (most reliable)
        # 2. Provided feature names
        # 3. None (will be handled later)
        if model_feature_names:
            self.feature_names = model_feature_names
            print(f"[DEBUG] ThreadSafeModelWrapper: Using model's feature names ({len(self.feature_names)} features)")
        elif feature_names:
            self.feature_names = list(feature_names)
            print(f"[DEBUG] ThreadSafeModelWrapper: Using provided feature names ({len(self.feature_names)} features)")
            # **WARNING**: Validate against model expectations if possible
            if hasattr(self.model, 'feature_names_in_'):
                expected_names = list(self.model.feature_names_in_)
                if set(self.feature_names) != set(expected_names):
                    print(f"[WARNING] ThreadSafeModelWrapper: Feature name mismatch detected!")
                    print(f"[WARNING] Provided: {self.feature_names[:3]}...")
                    print(f"[WARNING] Expected: {expected_names[:3]}...")
                    # **AUTO-FIX**: Use model's expected names instead
                    self.feature_names = expected_names
                    print(f"[FIX] ThreadSafeModelWrapper: Auto-corrected to use model's expected feature names")
        else:
            self.feature_names = None
            print(f"[DEBUG] ThreadSafeModelWrapper: No feature names provided")
        
        # Determine model capabilities
        self._has_predict_proba = self._check_predict_proba_support()
        
        # Initialize bounds info (will be set later if needed)
        self.bounds_info = None
        
        # **NEW FEATURE**: Extract preprocessor for inverse transform
        self.preprocessor = self._extract_preprocessor()
        
        # Store original feature bounds for proper inverse transform
        self.original_feature_bounds = None
        self.preprocessed_sample = None
    
    def set_feature_bounds_info(self, X_sample):
        """Store original feature ranges for proper inverse transform
        
        Args:
            X_sample: Sample of original (pre-processed) data to determine ranges
        """
        try:
            if X_sample is not None and len(X_sample) > 0:
                import pandas as pd
                
                if isinstance(X_sample, pd.DataFrame):
                    # Store min/max for each feature
                    self.original_feature_bounds = {
                        col: {'min': X_sample[col].min(), 'max': X_sample[col].max()}
                        for col in X_sample.columns
                    }
                    print(f"Debug: Stored original feature bounds for {len(self.original_feature_bounds)} features")
                    for col, bounds in list(self.original_feature_bounds.items())[:3]:  # Show first 3
                        print(f"  {col}: {bounds['min']:.6f} to {bounds['max']:.6f}")
                elif isinstance(X_sample, np.ndarray) and self.feature_names:
                    # Create DataFrame and store bounds
                    df = pd.DataFrame(X_sample, columns=self.feature_names)
                    self.original_feature_bounds = {
                        col: {'min': df[col].min(), 'max': df[col].max()}
                        for col in df.columns
                    }
                    print(f"Debug: Stored original feature bounds for {len(self.original_feature_bounds)} features")
                    
        except Exception as e:
            print(f"Debug: Failed to store feature bounds: {e}")
            self.original_feature_bounds = None
    
    def _check_predict_proba_support(self):
        """Check if the underlying model actually supports predict_proba"""
        try:
            # For Pipeline models, check the final estimator
            if hasattr(self.model, 'named_steps'):
                # Get the final estimator (usually the last step that's not a preprocessor)
                final_estimator = None
                for step_name, step in self.model.named_steps.items():
                    if step_name not in ['preprocessor', 'scaler', 'imputer', 'encoder']:
                        final_estimator = step
                
                if final_estimator is not None:
                    return hasattr(final_estimator, 'predict_proba')
            
            # For regular models, check directly
            return hasattr(self.model, 'predict_proba')
        except:
            return False
    
    def _extract_preprocessor(self):
        """Extract preprocessor from Pipeline for inverse transform"""
        try:
            # For Pipeline models, get the preprocessor
            if hasattr(self.model, 'named_steps'):
                for step_name, step in self.model.named_steps.items():
                    if step_name in ['preprocessor', 'scaler', 'imputer', 'encoder']:
                        return step
            
            # No preprocessor found
            return None
        except:
            return None
    
    def inverse_transform_features(self, X_optimizer):
        """Return optimizer values as-is since they are already in original scale
        
        Args:
            X_optimizer: Features in optimizer space (original scale), shape (n_samples, n_features)
            
        Returns:
            Features in original scale (same as input since bounds are now set correctly)
        """
        try:
            print(f"Debug: Optimizer values are already in original scale")
            print(f"Debug: Input shape: {X_optimizer.shape if hasattr(X_optimizer, 'shape') else 'no shape'}")
            print(f"Debug: Input sample values (original scale): {X_optimizer.flatten() if hasattr(X_optimizer, 'flatten') else X_optimizer}")
            
            # Convert to DataFrame if needed  
            if isinstance(X_optimizer, np.ndarray) and self.feature_names:
                import pandas as pd
                X_df = pd.DataFrame(X_optimizer, columns=self.feature_names)
            else:
                X_df = X_optimizer
            
            # Since bounds are now set to original scale, no transformation needed
            print(f"Debug: Returning values as-is (original scale)")
            return X_df
                
        except Exception as e:
            print(f"Debug: Feature display failed: {e}")
            return X_optimizer
    
    def _simple_bounds_inverse_transform(self, X_optimizer):
        """Simple inverse transform using stored original feature bounds"""
        try:
            import pandas as pd
            
            result_dict = {}
            
            for col in X_optimizer.columns:
                if col in self.original_feature_bounds:
                    bounds = self.original_feature_bounds[col]
                    min_val = bounds['min']
                    max_val = bounds['max']
                    
                    # Convert [0,1] optimizer values to original range
                    original_values = X_optimizer[col] * (max_val - min_val) + min_val
                    result_dict[col] = original_values
                    
                    print(f"Debug: {col}: {X_optimizer[col].iloc[0]:.6f} -> {original_values.iloc[0]:.6f} (range: {min_val:.6f} to {max_val:.6f})")
                else:
                    # No bounds info, keep as-is
                    result_dict[col] = X_optimizer[col]
                    print(f"Debug: {col}: No bounds info, keeping as-is: {X_optimizer[col].iloc[0]:.6f}")
            
            result_df = pd.DataFrame(result_dict)
            print(f"Debug: Simple bounds inverse transform completed")
            return result_df
            
        except Exception as e:
            print(f"Debug: Simple bounds inverse transform failed: {e}")
            return X_optimizer
    
    def _inverse_transform_step_by_step(self, X_optimizer):
        """Convert [0-1] optimizer values to original scale step by step"""
        try:
            import pandas as pd
            
            print(f"Debug: Step-by-step inverse transform")
            
            # The key insight: optimizer gives us [0-1] values
            # We need to reverse the entire preprocessing pipeline:
            # Original -> Preprocessed -> [0-1] (optimizer bounds)
            # We need: [0-1] (optimizer) -> Preprocessed -> Original
            
            # But there's a problem: we don't have access to the original feature bounds
            # that were used to create the [0-1] mapping
            
            # SIMPLE APPROACH: Assume optimizer [0-1] maps directly to preprocessed values
            # If preprocessor includes scaling, this won't work perfectly
            
            print(f"Debug: Attempting to treat optimizer values as preprocessed values")
            result_dict = {}
            
            if hasattr(self.preprocessor, 'transformers_'):
                for name, transformer, columns in self.preprocessor.transformers_:
                    if name == 'remainder' or transformer == 'drop':
                        continue
                    
                    print(f"Debug: Processing {name} transformer for columns {columns}")
                    
                    # Extract values for these columns
                    try:
                        X_subset = X_optimizer[columns]
                        print(f"Debug: Extracted values: {X_subset.iloc[0].values if hasattr(X_subset, 'iloc') else X_subset}")
                        
                        # Try inverse transform
                        if hasattr(transformer, 'inverse_transform') and transformer != 'passthrough':
                            try:
                                X_orig_subset = transformer.inverse_transform(X_subset)
                                print(f"Debug: Inverse transform successful for {name}")
                                print(f"Debug: Result: {X_orig_subset.flatten() if hasattr(X_orig_subset, 'flatten') else X_orig_subset}")
                                
                                # Store results
                                if isinstance(X_orig_subset, np.ndarray):
                                    for i, col in enumerate(columns):
                                        if i < X_orig_subset.shape[1]:
                                            result_dict[col] = X_orig_subset[:, i]
                                else:
                                    for col in columns:
                                        if col in X_orig_subset:
                                            result_dict[col] = X_orig_subset[col]
                                            
                            except Exception as e:
                                print(f"Debug: Inverse transform failed for {name}: {e}")
                                # Keep optimizer values as-is
                                for i, col in enumerate(columns):
                                    if isinstance(X_subset, pd.DataFrame) and i < len(X_subset.columns):
                                        result_dict[col] = X_subset.iloc[:, i].values
                        else:
                            print(f"Debug: No inverse transform for {name}, keeping values as-is")
                            # No inverse transform, keep as-is
                            for i, col in enumerate(columns):
                                if isinstance(X_subset, pd.DataFrame) and i < len(X_subset.columns):
                                    result_dict[col] = X_subset.iloc[:, i].values
                                    
                    except Exception as e:
                        print(f"Debug: Failed to process {name}: {e}")
                        # Use fallback values
                        for col in columns:
                            if col in X_optimizer:
                                result_dict[col] = X_optimizer[col]
                
                if result_dict:
                    result_df = pd.DataFrame(result_dict)
                    print(f"Debug: Final result: {result_df.iloc[0].values if hasattr(result_df, 'iloc') else result_df}")
                    return result_df
            
            # Fallback
            return X_optimizer
            
        except Exception as e:
            print(f"Debug: Step-by-step inverse transform failed: {e}")
            return X_optimizer
    
    def _manual_inverse_transform(self, X_transformed):
        """Manually reconstruct original features from transformed data"""
        try:
            import pandas as pd
            
            if hasattr(self.preprocessor, 'transformers_'):
                # ColumnTransformer case - we need to carefully handle the output structure
                result_dict = {}
                
                # Get the output feature names and their positions
                try:
                    output_feature_names = self.preprocessor.get_feature_names_out()
                except:
                    output_feature_names = None
                
                current_col_idx = 0
                
                for name, transformer, columns in self.preprocessor.transformers_:
                    if name == 'remainder' or transformer == 'drop':
                        continue
                    
                    # Determine how many columns this transformer produces
                    if hasattr(transformer, 'get_feature_names_out'):
                        try:
                            transformer_output_names = transformer.get_feature_names_out(columns)
                            n_output_cols = len(transformer_output_names)
                        except:
                            n_output_cols = len(columns)
                    else:
                        n_output_cols = len(columns)
                    
                    # Extract the columns for this transformer from the transformed data
                    # CRITICAL FIX: X_transformed has ORIGINAL column names, not output feature names
                    if isinstance(X_transformed, pd.DataFrame):
                        # Use original column names that belong to this transformer
                        try:
                            X_subset = X_transformed[columns]
                            print(f"Debug: Successfully extracted {len(columns)} columns for {name} transformer")
                        except KeyError as e:
                            print(f"Debug: KeyError extracting columns {columns}: {e}")
                            # Fallback to position-based extraction
                            X_subset = X_transformed.iloc[:, current_col_idx:current_col_idx + n_output_cols]
                    else:
                        # numpy array case
                        X_subset = X_transformed[:, current_col_idx:current_col_idx + n_output_cols]
                    
                    # Try to inverse transform this subset
                    try:
                        if hasattr(transformer, 'inverse_transform') and transformer != 'passthrough':
                            X_orig_subset = transformer.inverse_transform(X_subset)
                            
                            # Add to result - ensure we use original column names
                            if isinstance(X_orig_subset, np.ndarray):
                                for i, col in enumerate(columns):
                                    if i < X_orig_subset.shape[1]:
                                        result_dict[col] = X_orig_subset[:, i]
                                    else:
                                        # Fallback to transformed value if shape mismatch
                                        if isinstance(X_subset, pd.DataFrame) and i < len(X_subset.columns):
                                            result_dict[col] = X_subset.iloc[:, i].values
                                        elif isinstance(X_subset, np.ndarray) and i < X_subset.shape[1]:
                                            result_dict[col] = X_subset[:, i]
                            else:
                                # DataFrame result
                                for col in columns:
                                    if col in X_orig_subset:
                                        result_dict[col] = X_orig_subset[col]
                        else:
                            # No inverse transform (passthrough or similar), use original values
                            for i, col in enumerate(columns):
                                if isinstance(X_subset, pd.DataFrame) and i < len(X_subset.columns):
                                    result_dict[col] = X_subset.iloc[:, i].values
                                elif isinstance(X_subset, np.ndarray) and i < X_subset.shape[1]:
                                    result_dict[col] = X_subset[:, i]
                                
                    except Exception as e:
                        print(f"Debug: Failed to inverse transform {name} transformer: {e}")
                        # Failed for this transformer, use transformed values as-is
                        for i, col in enumerate(columns):
                            if isinstance(X_subset, pd.DataFrame) and i < len(X_subset.columns):
                                result_dict[col] = X_subset.iloc[:, i].values
                            elif isinstance(X_subset, np.ndarray) and i < X_subset.shape[1]:
                                result_dict[col] = X_subset[:, i]
                    
                    # Move to next set of columns
                    current_col_idx += n_output_cols
                
                # Return as DataFrame
                if result_dict:
                    # Ensure we maintain the original column order
                    if self.feature_names:
                        ordered_result = {}
                        for col in self.feature_names:
                            if col in result_dict:
                                ordered_result[col] = result_dict[col]
                        result_dict = ordered_result
                    
                    return pd.DataFrame(result_dict)
            
            # Fallback: return original
            return X_transformed
            
        except Exception as e:
            print(f"Debug: Manual inverse transform failed completely: {e}")
            # Ultimate fallback
            return X_transformed
    
    def predict(self, X):
        """Thread-safe predict method with enhanced feature name handling"""
        with self._lock:
            try:
                # Convert numpy array to DataFrame if needed for Pipeline compatibility
                if isinstance(X, np.ndarray) and self.feature_names:
                    import pandas as pd
                    
                    # **CRITICAL FIX**: Ensure feature names match exactly what the model expects
                    if len(self.feature_names) != X.shape[1]:
                        print(f"[WARNING] Feature count mismatch: Expected {len(self.feature_names)}, got {X.shape[1]}")
                        # Try to handle mismatch gracefully
                        if X.shape[1] < len(self.feature_names):
                            # Pad with zeros if we have fewer features than expected
                            padding = np.zeros((X.shape[0], len(self.feature_names) - X.shape[1]))
                            X = np.concatenate([X, padding], axis=1)
                        elif X.shape[1] > len(self.feature_names):
                            # Truncate if we have more features than expected
                            X = X[:, :len(self.feature_names)]
                    
                    # Create DataFrame with exact feature names the model expects
                    X_df = pd.DataFrame(X, columns=self.feature_names)
                    
                    # **DEBUG**: Print feature names for first few calls to help debugging
                    if not hasattr(self, '_debug_predict_count'):
                        self._debug_predict_count = 0
                    self._debug_predict_count += 1
                    
                    if self._debug_predict_count <= 2:  # Only print for first 2 calls
                        print(f"[DEBUG] Predict call #{self._debug_predict_count}")
                        print(f"[DEBUG] Input shape: {X.shape}")
                        print(f"[DEBUG] Feature names count: {len(self.feature_names)}")
                        print(f"[DEBUG] First few feature names: {self.feature_names[:3]}")
                        print(f"[DEBUG] DataFrame columns: {list(X_df.columns[:3])}")
                        
                        # Check if model has feature_names_in_ attribute
                        if hasattr(self.model, 'feature_names_in_'):
                            model_features = list(self.model.feature_names_in_)
                            print(f"[DEBUG] Model expects features: {model_features[:3]}")
                            
                            # Check for exact match
                            if set(self.feature_names) != set(model_features):
                                print(f"[WARNING] Feature name mismatch detected!")
                                print(f"[WARNING] Using features: {self.feature_names[:3]}...")
                                print(f"[WARNING] Model expects: {model_features[:3]}...")
                                
                                # **CRITICAL FIX**: Use model's expected feature names if available
                                if len(model_features) == X.shape[1]:
                                    print(f"[FIX] Updating DataFrame to use model's expected feature names")
                                    X_df = pd.DataFrame(X, columns=model_features)
                else:
                    X_df = X
                
                # **ENHANCED ERROR HANDLING**: Try prediction with detailed error reporting
                try:
                    result = self.model.predict(X_df)
                    return result
                except Exception as pred_error:
                    error_msg = str(pred_error)
                    print(f"[ERROR] Model prediction failed: {error_msg}")
                    
                    # **FEATURE NAME FIX**: If it's a feature name error, try to fix it
                    if "Unknown features" in error_msg or "feature_names_in_" in error_msg:
                        print(f"[FIX] Attempting to resolve feature name mismatch...")
                        
                        # Try to get the correct feature names from the model
                        try:
                            if hasattr(self.model, 'feature_names_in_'):
                                correct_features = list(self.model.feature_names_in_)
                                print(f"[FIX] Using model's feature_names_in_: {correct_features[:3]}...")
                                
                                if isinstance(X_df, pd.DataFrame) and len(correct_features) == len(X_df.columns):
                                    # Update the DataFrame with correct column names
                                    X_df_fixed = pd.DataFrame(X_df.values, columns=correct_features)
                                    result = self.model.predict(X_df_fixed)
                                    
                                    # **PERMANENT FIX**: Update our feature names for future calls
                                    self.feature_names = correct_features
                                    print(f"[FIX] Successfully fixed feature names. Prediction succeeded.")
                                    return result
                                    
                            # Try with pipeline steps if it's a Pipeline
                            elif hasattr(self.model, 'named_steps'):
                                print(f"[FIX] Trying to extract feature names from Pipeline...")
                                
                                # Check the first step (usually preprocessor)
                                first_step_name, first_step = list(self.model.named_steps.items())[0]
                                if hasattr(first_step, 'feature_names_in_'):
                                    correct_features = list(first_step.feature_names_in_)
                                    print(f"[FIX] Using {first_step_name} feature_names_in_: {correct_features[:3]}...")
                                    
                                    if isinstance(X_df, pd.DataFrame) and len(correct_features) == len(X_df.columns):
                                        X_df_fixed = pd.DataFrame(X_df.values, columns=correct_features)
                                        result = self.model.predict(X_df_fixed)
                                        
                                        # **PERMANENT FIX**: Update our feature names for future calls
                                        self.feature_names = correct_features
                                        print(f"[FIX] Successfully fixed feature names from Pipeline. Prediction succeeded.")
                                        return result
                                        
                        except Exception as fix_error:
                            print(f"[ERROR] Feature name fix attempt failed: {fix_error}")
                    
                    # If all fixes fail, raise the original error
                    raise pred_error
                    
            except Exception as e:
                print(f"[ERROR] Critical prediction error: {e}")
                # Return safe fallback values
                if hasattr(X, 'shape'):
                    return np.zeros(X.shape[0])
                else:
                    return np.array([0])
    
    def predict_proba(self, X):
        """Thread-safe predict_proba method with enhanced feature name handling - only available if underlying model supports it"""
        if not self._has_predict_proba:
            raise AttributeError("'ThreadSafeModelWrapper' object has no attribute 'predict_proba' - underlying model does not support it")
        
        with self._lock:
            try:
                # Convert numpy array to DataFrame if needed for Pipeline compatibility
                if isinstance(X, np.ndarray) and self.feature_names:
                    import pandas as pd
                    
                    # **CRITICAL FIX**: Ensure feature names match exactly what the model expects
                    if len(self.feature_names) != X.shape[1]:
                        print(f"[WARNING] Feature count mismatch in predict_proba: Expected {len(self.feature_names)}, got {X.shape[1]}")
                        # Try to handle mismatch gracefully
                        if X.shape[1] < len(self.feature_names):
                            # Pad with zeros if we have fewer features than expected
                            padding = np.zeros((X.shape[0], len(self.feature_names) - X.shape[1]))
                            X = np.concatenate([X, padding], axis=1)
                        elif X.shape[1] > len(self.feature_names):
                            # Truncate if we have more features than expected
                            X = X[:, :len(self.feature_names)]
                    
                    # Create DataFrame with exact feature names the model expects
                    X_df = pd.DataFrame(X, columns=self.feature_names)
                    
                    # **DEBUG**: Print feature names for first few calls to help debugging
                    if not hasattr(self, '_debug_predict_proba_count'):
                        self._debug_predict_proba_count = 0
                    self._debug_predict_proba_count += 1
                    
                    if self._debug_predict_proba_count <= 2:  # Only print for first 2 calls
                        print(f"[DEBUG] Predict_proba call #{self._debug_predict_proba_count}")
                        print(f"[DEBUG] Input shape: {X.shape}")
                        print(f"[DEBUG] Feature names count: {len(self.feature_names)}")
                        print(f"[DEBUG] First few feature names: {self.feature_names[:3]}")
                        
                        # Check if model has feature_names_in_ attribute
                        if hasattr(self.model, 'feature_names_in_'):
                            model_features = list(self.model.feature_names_in_)
                            print(f"[DEBUG] Model expects features: {model_features[:3]}")
                            
                            # Check for exact match
                            if set(self.feature_names) != set(model_features):
                                print(f"[WARNING] Feature name mismatch detected in predict_proba!")
                                
                                # **CRITICAL FIX**: Use model's expected feature names if available
                                if len(model_features) == X.shape[1]:
                                    print(f"[FIX] Updating DataFrame to use model's expected feature names")
                                    X_df = pd.DataFrame(X, columns=model_features)
                else:
                    X_df = X
                
                # **ENHANCED ERROR HANDLING**: Try prediction with detailed error reporting
                try:
                    result = self.model.predict_proba(X_df)
                    return result
                except Exception as pred_error:
                    error_msg = str(pred_error)
                    print(f"[ERROR] Model predict_proba failed: {error_msg}")
                    
                    # **FEATURE NAME FIX**: If it's a feature name error, try to fix it
                    if "Unknown features" in error_msg or "feature_names_in_" in error_msg:
                        print(f"[FIX] Attempting to resolve feature name mismatch in predict_proba...")
                        
                        # Try to get the correct feature names from the model
                        try:
                            if hasattr(self.model, 'feature_names_in_'):
                                correct_features = list(self.model.feature_names_in_)
                                print(f"[FIX] Using model's feature_names_in_: {correct_features[:3]}...")
                                
                                if isinstance(X_df, pd.DataFrame) and len(correct_features) == len(X_df.columns):
                                    # Update the DataFrame with correct column names
                                    X_df_fixed = pd.DataFrame(X_df.values, columns=correct_features)
                                    result = self.model.predict_proba(X_df_fixed)
                                    
                                    # **PERMANENT FIX**: Update our feature names for future calls
                                    self.feature_names = correct_features
                                    print(f"[FIX] Successfully fixed feature names for predict_proba. Prediction succeeded.")
                                    return result
                                    
                            # Try with pipeline steps if it's a Pipeline
                            elif hasattr(self.model, 'named_steps'):
                                print(f"[FIX] Trying to extract feature names from Pipeline for predict_proba...")
                                
                                # Check the first step (usually preprocessor)
                                first_step_name, first_step = list(self.model.named_steps.items())[0]
                                if hasattr(first_step, 'feature_names_in_'):
                                    correct_features = list(first_step.feature_names_in_)
                                    print(f"[FIX] Using {first_step_name} feature_names_in_: {correct_features[:3]}...")
                                    
                                    if isinstance(X_df, pd.DataFrame) and len(correct_features) == len(X_df.columns):
                                        X_df_fixed = pd.DataFrame(X_df.values, columns=correct_features)
                                        result = self.model.predict_proba(X_df_fixed)
                                        
                                        # **PERMANENT FIX**: Update our feature names for future calls
                                        self.feature_names = correct_features
                                        print(f"[FIX] Successfully fixed feature names from Pipeline for predict_proba. Prediction succeeded.")
                                        return result
                                        
                        except Exception as fix_error:
                            print(f"[ERROR] Feature name fix attempt failed for predict_proba: {fix_error}")
                    
                    # If all fixes fail, raise the original error
                    raise pred_error
                    
            except Exception as e:
                print(f"[ERROR] Critical predict_proba error: {e}")
                # Return safe fallback probabilities
                if hasattr(X, 'shape'):
                    n_samples = X.shape[0]
                    # Return binary classification probabilities [0.5, 0.5]
                    return np.full((n_samples, 2), 0.5)
                else:
                    return np.array([[0.5, 0.5]])
    
    def __getattr__(self, name):
        """Delegate other attributes to the wrapped model"""
        # Special handling for predict_proba to respect underlying model capabilities
        if name == 'predict_proba' and not self._has_predict_proba:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' - underlying model does not support it")
        return getattr(self.model, name)

class OptimizationWorker(QThread):
    """Enhanced optimization worker thread with thread-safe model access and SHAP conflict protection"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    iteration_completed = pyqtSignal(int, float, list, dict)  # iteration, best_value, best_params, extra_data
    optimization_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    real_time_data = pyqtSignal(dict)  # For real-time visualization updates
    
    def __init__(self, model, bounds_dict, constraints, algorithm, config, target_direction, feature_names=None):
        super().__init__()
        # Wrap model for thread safety
        self.model = ThreadSafeModelWrapper(model, feature_names)
        self.bounds_dict = bounds_dict if bounds_dict else {}
        self.constraints = constraints if constraints else []
        self.algorithm = algorithm
        self.config = config if config else {}
        self.target_direction = target_direction
        self.feature_names = feature_names if feature_names else list(self.bounds_dict.keys())
        self.should_stop = False
        self.history = []
        self.best_value = None
        self.best_params = None
        self.constraint_violations = 0
        
        # Enhanced tracking for real-time visualization
        self.real_time_history = []
        self.population_data = []  # For GA/PSO population tracking
        self.convergence_data = []
        
        # **CRITICAL FIX**: Determine model type once during initialization
        self.is_classifier = self._determine_model_type()
        
        # **CRITICAL FIX FOR CATEGORICAL FEATURES**: Store mappings between optimizer values and original categorical values
        # Format: {feature_name: {'orig_to_num': {original_val: numeric_val}, 'num_to_orig': {numeric_val: original_val}}}
        self.categorical_mappings = {}
        self._initialize_categorical_mappings()
    
    def _initialize_categorical_mappings(self):
        """Initialize categorical mappings for all categorical features"""
        for feature_name, bounds in self.bounds_dict.items():
            if bounds.feature_type == "categorical":
                self.categorical_mappings[feature_name] = {
                    'orig_to_num': {},
                    'num_to_orig': {}
                }
                for value in bounds.categorical_values:
                    self.categorical_mappings[feature_name]['orig_to_num'][value] = len(self.categorical_mappings[feature_name]['orig_to_num'])
                    self.categorical_mappings[feature_name]['num_to_orig'][len(self.categorical_mappings[feature_name]['num_to_orig'])] = value
    
    def _determine_model_type(self):
        """
        Determine if the model is a classifier or regressor by checking the actual model,
        not the wrapper. This is done once during initialization to avoid repeated checks.
        """
        try:
            # Get the actual model from the wrapper
            actual_model = self.model.model
            
            # Check if it's a Pipeline
            if hasattr(actual_model, 'named_steps'):
                # For Pipeline, check the final estimator
                final_estimator = None
                for step_name, step in actual_model.named_steps.items():
                    # Skip preprocessing steps
                    if step_name not in ['preprocessor', 'scaler', 'imputer', 'encoder']:
                        final_estimator = step
                        break
                
                if final_estimator is not None:
                    # Check if the final estimator has predict_proba method
                    return hasattr(final_estimator, 'predict_proba')
                else:
                    # Fallback: check the last step
                    last_step = list(actual_model.named_steps.values())[-1]
                    return hasattr(last_step, 'predict_proba')
            else:
                # For non-Pipeline models, check directly
                return hasattr(actual_model, 'predict_proba')
                
        except Exception as e:
            # If we can't determine the type, default to regressor (safer)
            print(f"[WARNING] Could not determine model type: {e}. Defaulting to regressor.")
            return False
    
    def run(self):
        """Run optimization algorithm with enhanced real-time tracking"""
        try:
            self.status_updated.emit(f"Starting {self.algorithm} optimization...")
            self.progress_updated.emit(0)
            
            # Prepare bounds and constraints for optimization
            bounds, space_config = self._prepare_bounds_and_space()
            
            def objective(x):
                """Enhanced objective function with robust categorical handling and constraint enforcement"""
                try:
                    # Convert to array if needed
                    if hasattr(x, '__iter__') and not isinstance(x, np.ndarray):
                        x = np.array(x)
                    
                    # Apply categorical constraints FIRST to ensure all evaluations use valid discrete values
                    # This is critical for consistent constraint evaluation with categorical features
                    x_constrained = self._apply_categorical_constraints(x)
                    
                    # **CRITICAL FIX FOR CATEGORICAL FEATURES**: Convert numerical values back to original categorical values
                    # Create a copy that will hold mixed types (numbers and original categorical values)
                    x_for_model = self._convert_to_original_categories(x_constrained)
                    
                    # Setup debugging
                    if not hasattr(self, '_debug_call_count'):
                        self._debug_call_count = 0
                    self._debug_call_count += 1
                    
                    debug_output = self._debug_call_count <= 5  # Show more debug info for first 5 calls
                    
                    if debug_output:
                        print(f"\n[OBJECTIVE] Call #{self._debug_call_count}")
                        print(f"[OBJECTIVE] Input parameters: {x_constrained[:5]}...")
                        
                        # Check for categorical features and highlight them
                        if hasattr(self, 'bounds_dict') and self.feature_names:
                            categorical_features = []
                            for i, feature_name in enumerate(self.feature_names[:10]):  # First 10 features
                                if i < len(x_constrained) and feature_name in self.bounds_dict:
                                    bounds = self.bounds_dict[feature_name]
                                    if bounds and bounds.feature_type in ["categorical", "binary"]:
                                        orig_value = x_for_model[i] if i < len(x_for_model) else "unknown"
                                        categorical_features.append(f"{feature_name}({i})={x_constrained[i]}->{orig_value}")
                            
                            if categorical_features:
                                print(f"[OBJECTIVE] Categorical features: {', '.join(categorical_features)}")
                        
                        print(f"[OBJECTIVE] Model type: {type(self.model)}")
                    
                    # Use pre-determined model type for prediction
                    try:
                        # Create a DataFrame with proper feature names and mixed types
                        import pandas as pd
                        x_df = pd.DataFrame([x_for_model], columns=self.feature_names)
                        
                        if debug_output:
                            print(f"[OBJECTIVE] DataFrame for prediction: {x_df.dtypes.to_dict()}")
                            for col in x_df.columns:
                                if col in self.categorical_mappings:
                                    print(f"[OBJECTIVE] Categorical column {col}: {x_df[col].values[0]}")
                        
                        if self.is_classifier:
                            # Classification mode: use predict_proba
                            if debug_output:
                                print(f"[OBJECTIVE] Using CLASSIFICATION mode (predict_proba)")
                            
                            proba = self.model.predict_proba(x_df)[0]
                            
                            # For binary classification, use probability of positive class (index 1)
                            if len(proba) == 2:
                                prediction = proba[1]  # Probability of class 1
                                if debug_output:
                                    print(f"[OBJECTIVE] Binary classification: positive class probability = {prediction:.6f}")
                            else:
                                # Multi-class: use max probability
                                prediction = np.max(proba)
                                if debug_output:
                                    print(f"[OBJECTIVE] Multi-class: max probability = {prediction:.6f}")
                        else:
                            # Regression mode: use predict
                            if debug_output:
                                print(f"[OBJECTIVE] Using REGRESSION mode (predict)")
                            
                            prediction = self.model.predict(x_df)[0]
                            if debug_output:
                                print(f"[OBJECTIVE] Regression prediction: {prediction:.6f}")
                                
                    except Exception as pred_error:
                        print(f"[OBJECTIVE ERROR] Model prediction failed: {str(pred_error)}")
                        traceback.print_exc()
                        return float('inf') if self.target_direction == "minimize" else -float('inf')
                    
                    # Store current prediction for adaptive constraint scaling
                    self.last_objective_value = prediction
                    
                    # Update objective scale estimate for constraint penalties
                    if not self.config.get('objective_scale_estimate'):
                        self.config['objective_scale_estimate'] = abs(prediction)
                    else:
                        # Exponential moving average for scale estimate
                        current = self.config['objective_scale_estimate']
                        self.config['objective_scale_estimate'] = 0.9 * current + 0.1 * abs(prediction)
                    
                    if debug_output:
                        print(f"[OBJECTIVE] Raw prediction: {prediction:.6f}")
                        print(f"[OBJECTIVE] Objective scale estimate: {self.config['objective_scale_estimate']:.6f}")
                    
                    # For COBYLA with explicit constraints, return raw objective
                    # COBYLA handles constraints through separate constraint functions
                    if self.algorithm == "COBYLA" and hasattr(self, 'constraints') and self.constraints:
                        final_objective = -prediction if self.target_direction == "maximize" else prediction
                        
                        if debug_output:
                            print(f"[OBJECTIVE] COBYLA objective (constraints handled separately): {final_objective:.6f}")
                            
                        return final_objective
                    
                    # For non-COBYLA algorithms, apply penalty method with enhanced scaling
                    if hasattr(self, 'constraints') and self.constraints:
                        # Evaluate constraints on the constrained vector (critical for categorical features)
                        constraint_penalty = self._evaluate_constraints(x_constrained)
                        
                        if constraint_penalty > 0:
                            self.constraint_violations += 1
                            
                            # Apply penalty correctly based on optimization direction
                            if self.target_direction == "maximize":
                                # For maximization, we need to make the objective worse (more negative)
                                # So we subtract the penalty from the prediction
                                adjusted_prediction = prediction - constraint_penalty
                            else:
                                # For minimization, we need to make the objective worse (more positive)
                                # So we add the penalty to the prediction
                                adjusted_prediction = prediction + constraint_penalty
                                
                            if debug_output:
                                print(f"[OBJECTIVE] Constraint penalty: {constraint_penalty:.6f}")
                                print(f"[OBJECTIVE] Prediction adjusted from {prediction:.6f} to {adjusted_prediction:.6f}")
                                
                            prediction = adjusted_prediction
                    
                    # Return negative for maximization (scipy minimizes)
                    final_objective = -prediction if self.target_direction == "maximize" else prediction
                    
                    # Store for debugging
                    if not hasattr(self, 'last_objective'):
                        self.last_objective = []
                    self.last_objective.append(final_objective)
                    
                    if debug_output:
                        print(f"[OBJECTIVE] Final objective: {final_objective:.6f}")
                        print(f"[OBJECTIVE] Target direction: {self.target_direction}")
                        print("=" * 50)
                        
                    return final_objective
                    
                except Exception as e:
                    print(f"[OBJECTIVE ERROR] Objective function error: {str(e)}")
                    traceback.print_exc()
                    # Return a large value for minimization, small for maximization
                    return float('inf') if self.target_direction == "minimize" else -float('inf')
            
            # Enhanced callback for real-time updates with reduced frequency
            def optimization_callback(iteration, best_value, best_params, extra_data=None):
                """Callback function for real-time optimization updates with anti-overflow protection and categorical constraint fixing"""
                if self.should_stop:
                    return True
                
                # Convert objective value for display
                display_value = -best_value if self.target_direction == "maximize" else best_value
                
                # CRITICAL FIX: Apply categorical constraints to best_params before storing/emitting
                if best_params is not None:
                    best_params_fixed = self._apply_categorical_constraints(np.array(best_params))
                else:
                    best_params_fixed = None
                
                # Update tracking data (always track) - use fixed parameters
                self.real_time_history.append({
                    'iteration': iteration,
                    'best_value': display_value,
                    'best_params': best_params_fixed.copy() if best_params_fixed is not None else None,
                    'timestamp': time.time(),
                    'extra_data': extra_data if extra_data else {}
                })
                
                # Store population data for GA/PSO (limit storage to prevent memory issues)
                if extra_data and 'population_fitness' in extra_data:
                    if len(self.population_data) < 1000:  # Limit to 1000 entries
                        self.population_data.append(extra_data)
                
                # **CRITICAL FIX**: Reduce signal emission frequency to prevent stack overflow
                # Only emit signals every 5 iterations to reduce UI load
                should_emit = (iteration % 5 == 0) or (iteration < 10) or (len(self.real_time_history) % 25 == 0)
                
                if should_emit:
                    try:
                        # Emit signals for UI updates with error protection - use fixed parameters
                        self.iteration_completed.emit(iteration, display_value, 
                                                    best_params_fixed.tolist() if best_params_fixed is not None else [], 
                                                    extra_data if extra_data else {})
                        
                        # Emit real-time data for advanced visualization (reduced payload)
                        self.real_time_data.emit({
                            'iteration': iteration,
                            'best_value': display_value,
                            'history': self.real_time_history[-25:],  # Reduced from 50 to 25 points
                            'extra_data': extra_data if extra_data else {}
                        })
                        
                        # Update progress (rough estimate)
                        max_iterations = self.config.get('max_iterations', 100)
                        progress = min(90, int(iteration / max_iterations * 90))
                        self.progress_updated.emit(progress)
                        
                    except Exception as e:
                        # Silently handle signal emission errors to prevent crashes
                        pass
                
                return False
            
            # Prepare scipy constraints if available
            scipy_constraints = self._prepare_scipy_constraints() if SCIPY_AVAILABLE else []
            
            # Validate bounds before optimization
            print(f"[DEBUG] Algorithm: {self.algorithm}")
            print(f"[DEBUG] Bounds validation:")
            for i, (min_val, max_val) in enumerate(bounds):
                if min_val > max_val:
                    error_msg = f"Invalid bounds for feature {i}: min={min_val} > max={max_val}"
                    print(f"[ERROR] {error_msg}")
                    raise ValueError(error_msg)
                elif min_val == max_val:
                    print(f"[INFO] Fixed value for feature {i}: {min_val}")
                else:
                    print(f"[INFO] Feature {i} bounds: [{min_val}, {max_val}]")
            
            # Run selected algorithm with enhanced implementations
            start_time = time.time()
            
            if self.algorithm == "Genetic Algorithm":
                result = self._genetic_algorithm(objective, bounds, optimization_callback)
            elif self.algorithm == "Particle Swarm Optimization":
                result = self._particle_swarm_optimization(objective, bounds, optimization_callback)
            elif self.algorithm == "Grid Search":
                result = self._grid_search(objective, bounds, self.feature_names)
            elif self.algorithm == "Random Search":
                result = self._random_search(objective, bounds, self.feature_names)
            elif self.algorithm == "Differential Evolution" and SCIPY_AVAILABLE:
                result = self._differential_evolution(objective, bounds, scipy_constraints, optimization_callback)
            elif self.algorithm == "Basin Hopping" and SCIPY_AVAILABLE:
                result = self._basin_hopping(objective, bounds, scipy_constraints, optimization_callback)
            elif self.algorithm == "COBYLA" and SCIPY_AVAILABLE:
                result = self._cobyla_optimization(objective, bounds, scipy_constraints, optimization_callback)
            elif self.algorithm == "Bayesian Optimization" and SKOPT_AVAILABLE:
                # For Bayesian optimization, validate space_config
                print(f"[DEBUG] Space config validation for Bayesian Optimization:")
                for i, space in enumerate(space_config):
                    print(f"[INFO] Space {i}: {space}")
                result = self._bayesian_optimization(objective, space_config, optimization_callback)
            else:
                raise ValueError(f"Algorithm {self.algorithm} not available")
            
            execution_time = time.time() - start_time
            
            if not self.should_stop and result is not None:
                self.progress_updated.emit(100)
                self.status_updated.emit("Optimization completed successfully!")
                
                # Enhanced result validation and error recovery
                if 'x' not in result or result['x'] is None:
                    # Try to create a fallback result from real-time history
                    if self.real_time_history:
                        last_entry = self.real_time_history[-1]
                        if last_entry.get('best_params') is not None:
                            result['x'] = np.array(last_entry['best_params'])
                        else:
                            # Create random fallback within bounds
                            bounds, _ = self._prepare_bounds_and_space()
                            fallback_x = []
                            for min_val, max_val in bounds:
                                if min_val == max_val:
                                    fallback_x.append(min_val)
                                else:
                                    fallback_x.append(np.random.uniform(min_val, max_val))
                            result['x'] = np.array(fallback_x)
                    else:
                        raise ValueError("Optimization result missing 'x' (best parameters) and no fallback available")
                
                if 'fun' not in result:
                    result['fun'] = float('inf')
                
                # Validate that result['x'] is a proper array-like object
                try:
                    result['x'] = np.array(result['x'])
                    if result['x'].size == 0:
                        raise ValueError("Empty parameter array")
                except Exception as e:
                    raise ValueError(f"Invalid parameter array in result: {e}")
                    
                # Apply final categorical constraints to result
                result['x'] = self._apply_categorical_constraints(result['x'])
                
                # **NEW FEATURE**: Add model wrapper for inverse transform
                # Prepare enhanced results with performance metrics
                results = {
                    'algorithm': self.algorithm,
                    'best_params': dict(zip(self.feature_names, result['x'])),
                    'best_value': -result['fun'] if self.target_direction == "maximize" else result['fun'],
                    'history': self.history,
                    'real_time_history': self.real_time_history,
                    'population_data': self.population_data,
                    'feature_names': self.feature_names,
                    'target_direction': self.target_direction,
                    'n_iterations': len(self.real_time_history),
                    'constraint_violations': self.constraint_violations,
                    'constraints_satisfied': self._evaluate_constraints(result['x']) == 0,
                    'execution_time': execution_time,
                    'evaluations_per_second': len(self.real_time_history) / execution_time if execution_time > 0 else 0,
                    'convergence_data': self.convergence_data,
                    'final_objective_value': result['fun'],
                    'algorithm_specific_data': result.get('algorithm_data', {}),
                    'model_wrapper': self.model  # Add model wrapper for inverse transform
                }
                
                self.optimization_completed.emit(results)
            elif self.should_stop:
                self.status_updated.emit("Optimization stopped by user")
            else:
                raise ValueError("Optimization returned no results")
            
        except Exception as e:
            self.error_occurred.emit(f"Optimization error: {str(e)}\n{traceback.format_exc()}")
    
    def stop_optimization(self):
        """Stop optimization with cleanup"""
        self.should_stop = True
        
        # Stop algorithm-specific optimizers
        if hasattr(self, 'ga_optimizer'):
            self.ga_optimizer.should_stop = True
        if hasattr(self, 'pso_optimizer'):
            self.pso_optimizer.should_stop = True
    
    def _genetic_algorithm(self, objective, bounds, callback):
        """Enhanced Genetic Algorithm implementation with hybrid feature type support"""
        config = self.config
        
        # **HYBRID GA ENHANCEMENT**: Extract feature types from bounds_dict
        feature_types = []
        for feature_name in self.feature_names:
            if feature_name in self.bounds_dict:
                bounds_obj = self.bounds_dict[feature_name]
                if bounds_obj and hasattr(bounds_obj, 'feature_type'):
                    feature_types.append(bounds_obj.feature_type)
                else:
                    feature_types.append('continuous')
            else:
                feature_types.append('continuous')
        
        print(f"[HYBRID GA] Feature types for optimization: {feature_types}")
        
        self.ga_optimizer = GeneticAlgorithm(
            objective_func=objective,
            bounds=bounds,
            population_size=config.get('population_size', 50),
            max_generations=config.get('max_generations', 100),
            mutation_rate=config.get('mutation_rate', 0.1),
            crossover_rate=config.get('crossover_rate', 0.8),
            elite_size=config.get('elite_size', 5),
            callback=callback,
            feature_types=feature_types  # Pass feature types to hybrid GA
        )
        
        self.ga_optimizer.should_stop = False
        result = self.ga_optimizer.optimize()
        
        # Add algorithm-specific data
        result['algorithm_data'] = {
            'generation_history': self.ga_optimizer.generation_history,
            'best_fitness_history': self.ga_optimizer.best_fitness_history,
            'population_diversity': self.ga_optimizer.population_diversity
        }
        
        return result
    
    def _particle_swarm_optimization(self, objective, bounds, callback):
        """Enhanced Particle Swarm Optimization implementation with hybrid feature type support"""
        config = self.config
        
        # **HYBRID PSO ENHANCEMENT**: Extract feature types from bounds_dict
        feature_types = []
        for feature_name in self.feature_names:
            if feature_name in self.bounds_dict:
                bounds_obj = self.bounds_dict[feature_name]
                if bounds_obj and hasattr(bounds_obj, 'feature_type'):
                    feature_types.append(bounds_obj.feature_type)
                else:
                    feature_types.append('continuous')
            else:
                feature_types.append('continuous')
        
        print(f"[HYBRID PSO] Feature types for optimization: {feature_types}")
        
        self.pso_optimizer = ParticleSwarmOptimization(
            objective_func=objective,
            bounds=bounds,
            n_particles=config.get('n_particles', 30),
            max_iterations=config.get('max_iterations', 100),
            w=config.get('inertia_weight', 0.9),
            c1=config.get('cognitive_param', 2.0),
            c2=config.get('social_param', 2.0),
            callback=callback,
            feature_types=feature_types  # Pass feature types to hybrid PSO
        )
        
        self.pso_optimizer.should_stop = False
        result = self.pso_optimizer.optimize()
        
        # Add algorithm-specific data
        result['algorithm_data'] = {
            'iteration_history': self.pso_optimizer.iteration_history,
            'best_fitness_history': self.pso_optimizer.best_fitness_history,
            'swarm_positions': self.pso_optimizer.swarm_positions,
            'velocities_magnitude': self.pso_optimizer.velocities_magnitude
        }
        
        return result

    def _apply_categorical_constraints(self, x):
        """Apply categorical and integer constraints to parameter vector with enhanced debugging
        
        CRITICAL FIX: Use self.feature_names as the authoritative source of feature order
        rather than relying on dictionary iteration order which is not guaranteed to be consistent.
        
        ENHANCED: Handle One-Hot encoded categorical features with mutual exclusion constraints
        """
        x_constrained = x.copy()
        
        # Only apply constraints if we have bounds_dict and feature types are properly set
        if not hasattr(self, 'bounds_dict') or not self.bounds_dict:
            return x_constrained
            
        # Only apply constraints if we have feature names
        if not self.feature_names:
            print("[WARNING] No feature_names available for categorical constraints")
            return x_constrained
        
        # CRITICAL FIX: Detect and handle One-Hot encoded categorical features
        # Group features by their base name (before the underscore)
        one_hot_groups = {}
        for i, feature_name in enumerate(self.feature_names):
            # Check if this looks like a one-hot encoded feature (contains underscore and has binary bounds)
            if '_' in feature_name and i < len(x_constrained):
                base_name = feature_name.rsplit('_', 1)[0]  # Get everything before the last underscore
                bounds = self.bounds_dict.get(feature_name)
                
                # Only group if it's a binary feature (likely from one-hot encoding)
                if bounds and (bounds.feature_type == "binary" or 
                              (bounds.feature_type == "categorical" and 
                               bounds.categorical_values and 
                               set(bounds.categorical_values) == {0, 1})):
                    if base_name not in one_hot_groups:
                        one_hot_groups[base_name] = []
                    one_hot_groups[base_name].append(i)
        
        # Apply One-Hot constraints for grouped features
        for base_name, indices in one_hot_groups.items():
            if len(indices) > 1:  # Only apply if there are multiple features in the group
                # Get values for this group
                group_values = [x_constrained[i] for i in indices]
                
                # Find the index with the highest value (winner-takes-all)
                max_idx = np.argmax(group_values)
                
                # Set the winner to 1 and others to 0
                for j, idx in enumerate(indices):
                    x_constrained[idx] = 1.0 if j == max_idx else 0.0
        
        # Debug output for first few calls
        if not hasattr(self, '_categorical_debug_count'):
            self._categorical_debug_count = 0
        self._categorical_debug_count += 1
        
        debug_output = self._categorical_debug_count <= 3
        
        if debug_output:
            print(f"\n[CATEGORICAL] Processing vector of length {len(x_constrained)}")
            print(f"[CATEGORICAL] Feature bounds dictionary has {len(self.bounds_dict)} entries")
            print(f"[CATEGORICAL] Using {len(self.feature_names)} feature names as authoritative order")
            
            # Print One-Hot groups detected
            if one_hot_groups:
                print(f"[CATEGORICAL] One-Hot groups detected: {len(one_hot_groups)}")
                for base_name, indices in one_hot_groups.items():
                    if len(indices) > 1:
                        feature_names_in_group = [self.feature_names[i] for i in indices]
                        values_before = [x[i] for i in indices]
                        values_after = [x_constrained[i] for i in indices]
                        print(f"   Group '{base_name}': {feature_names_in_group}")
                        print(f"     Before: {[f'{v:.3f}' for v in values_before]}")
                        print(f"     After:  {[f'{v:.3f}' for v in values_after]}")
            
            # Print feature mapping for debugging
            print("[CATEGORICAL] Feature index mapping:")
            for i, feature_name in enumerate(self.feature_names):
                if i < len(x_constrained):
                    bounds = self.bounds_dict.get(feature_name)
                    print(f"   {i}: {feature_name} -> {bounds.feature_type if bounds else 'None'}")
        
        # Track changes for debugging
        changes_made = []
        
        # CRITICAL FIX: Process features in the exact order specified by self.feature_names
        for i, feature_name in enumerate(self.feature_names):
            # Skip if index out of bounds
            if i >= len(x_constrained):
                if debug_output:
                    print(f"[CATEGORICAL WARNING] Feature index {i} ({feature_name}) is out of bounds for vector of length {len(x_constrained)}")
                continue
                
            # Skip if feature not in bounds_dict
            if feature_name not in self.bounds_dict:
                if debug_output:
                    print(f"[CATEGORICAL WARNING] Feature {feature_name} not found in bounds_dict")
                continue
                
            # Get bounds for this feature
            bounds = self.bounds_dict[feature_name]
            if bounds is None:
                continue
                
            original_value = x_constrained[i]
            
            # Apply constraints based on feature type with improved handling
            if bounds.feature_type == "categorical" and bounds.categorical_values:
                # For categorical features, find the closest allowed value
                if len(bounds.categorical_values) > 0:
                    distances = [abs(original_value - val) for val in bounds.categorical_values]
                    closest_idx = np.argmin(distances)
                    x_constrained[i] = bounds.categorical_values[closest_idx]
                    
                    if debug_output:
                        changes_made.append(f"Feature {i} ({feature_name}): {original_value:.4f} -> {x_constrained[i]:.4f} (categorical)")
                
            elif bounds.feature_type == "binary":
                # For binary features, use strict threshold at 0.5
                if original_value >= 0.5:
                    x_constrained[i] = 1.0
                else:
                    x_constrained[i] = 0.0
                
                if debug_output and x_constrained[i] != original_value:
                    changes_made.append(f"Feature {i} ({feature_name}): {original_value:.4f} -> {x_constrained[i]:.4f} (binary)")
                    
            elif bounds.feature_type == "integer":
                # For integer features, round to nearest integer and ensure within bounds
                rounded_val = round(original_value)
                
                # Ensure within integer bounds if specified
                if hasattr(bounds, 'min_value') and hasattr(bounds, 'max_value'):
                    min_val = int(bounds.min_value)
                    max_val = int(bounds.max_value)
                    rounded_val = max(min_val, min(rounded_val, max_val))
                
                x_constrained[i] = rounded_val
                
                if debug_output and x_constrained[i] != original_value:
                    changes_made.append(f"Feature {i} ({feature_name}): {original_value:.4f} -> {x_constrained[i]:.4f} (integer)")
        
        # Print summary of changes for debugging
        if debug_output:
            if changes_made:
                print("[CATEGORICAL] Changes applied:")
                for change in changes_made:
                    print(f"   {change}")
                print(f"[CATEGORICAL] Total: {len(changes_made)} changes out of {len(x_constrained)} features")
            else:
                print("[CATEGORICAL] No changes needed - all features already satisfy type constraints")
            
        return x_constrained
    
    def _evaluate_constraints(self, x):
        """Enhanced constraint evaluation with robust feature mapping and extreme penalties"""
        if not self.constraints:
            return 0.0
            
        total_penalty = 0.0
        violations_count = 0
        
        # Debug flag for detailed output
        debug_output = not hasattr(self, '_constraints_debug_count') or self._constraints_debug_count < 5
        if not hasattr(self, '_constraints_debug_count'):
            self._constraints_debug_count = 0
        self._constraints_debug_count += 1
        
        # First, apply categorical constraints to ensure we're evaluating with proper values
        # This is critical for categorical features to ensure constraints are evaluated on valid values
        x_constrained = self._apply_categorical_constraints(x)
        
        # Create robust bidirectional feature name-index mappings
        feature_index_to_name = {}
        feature_name_to_index = {}
        
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                if i < len(x_constrained):  # Only map features within vector length
                    feature_index_to_name[i] = name
                    feature_name_to_index[name] = i
        
        # Get feature types for better debugging
        feature_types = {}
        if hasattr(self, 'bounds_dict') and self.bounds_dict:
            for feature_name, bounds in self.bounds_dict.items():
                if bounds and hasattr(bounds, 'feature_type'):
                    feature_types[feature_name] = bounds.feature_type
        
        # Print feature mapping once for debugging
        if debug_output:
            print(f"\n[CONSTRAINT] Evaluating {len(self.constraints)} constraints on vector of length {len(x_constrained)}")
            print(f"[CONSTRAINT] Feature mapping sample (first 5 features):")
            for i, name in list(feature_index_to_name.items())[:5]:
                feature_type = feature_types.get(name, "unknown")
                print(f"   - Index {i}: {name} ({feature_type})")
        
        # Track all violations for detailed reporting
        all_violations = []
        
        # Process each constraint
        for i, constraint in enumerate(self.constraints):
            try:
                # Calculate constraint value with improved feature mapping
                constraint_value = 0.0
                feature_values = []
                invalid_indices = []
                
                # Verify all indices are valid before proceeding
                for idx in constraint.feature_indices:
                    if idx >= len(x_constrained):
                        invalid_indices.append(idx)
                
                if invalid_indices:
                    print(f"[CONSTRAINT ERROR] Constraint {i+1} ({constraint.description}) references invalid feature indices: {invalid_indices}")
                    print(f"[CONSTRAINT ERROR] Vector length is {len(x_constrained)}, but constraint uses indices up to {max(constraint.feature_indices)}")
                    # Add extreme penalty for invalid constraints
                    total_penalty += 100000.0
                    violations_count += 1
                    continue
                
                # Calculate constraint value using constrained feature values
                for idx, coeff in zip(constraint.feature_indices, constraint.coefficients):
                    feature_value = x_constrained[idx]  # Use constrained values
                    constraint_value += coeff * feature_value
                    
                    # Get feature name and type for better debugging
                    feature_name = feature_index_to_name.get(idx, f"unknown_{idx}")
                    feature_type = feature_types.get(feature_name, "unknown")
                    
                    # Store for detailed debugging
                    feature_values.append((idx, feature_name, feature_type, feature_value, coeff))
                
                # Check constraint violation with improved tolerance handling
                violation = 0.0
                tolerance = 1e-6  # Base tolerance for floating point comparisons
                
                # Adjust tolerance for categorical features
                has_categorical = any(feature_types.get(feature_index_to_name.get(idx, ""), "") in ["categorical", "binary"] 
                                     for idx in constraint.feature_indices)
                
                if has_categorical:
                    # Use stricter tolerance for categorical features
                    tolerance = 1e-10
                
                if constraint.operator == "<=":
                    violation = max(0, constraint_value - constraint.bound)
                    constraint_status = "SATISFIED" if violation <= tolerance else "VIOLATED"
                elif constraint.operator == ">=":
                    violation = max(0, constraint.bound - constraint_value)
                    constraint_status = "SATISFIED" if violation <= tolerance else "VIOLATED"
                elif constraint.operator == "==":
                    violation = abs(constraint_value - constraint.bound)
                    constraint_status = "SATISFIED" if violation <= tolerance else "VIOLATED"
                else:
                    print(f"[CONSTRAINT ERROR] Unknown operator: {constraint.operator}")
                    violation = 1.0  # Default violation
                    constraint_status = "ERROR"
                
                # Print constraint evaluation details for debugging
                if debug_output or violation > tolerance:
                    print(f"\n[CONSTRAINT {i+1}] {constraint.description}")
                    print(f"[CONSTRAINT {i+1}] Feature values used:")
                    for idx, name, type_name, value, coeff in feature_values:
                        print(f"   - Feature {idx} ({name}, {type_name}): {value:.6f} × {coeff:.6f} = {value * coeff:.6f}")
                    print(f"[CONSTRAINT {i+1}] Total value: {constraint_value:.6f} {constraint.operator} {constraint.bound:.6f}")
                    print(f"[CONSTRAINT {i+1}] Status: {constraint_status}, Violation: {violation:.6f}")
                
                # Apply EXTREME penalty for violations with progressive scaling
                if violation > tolerance:
                    violations_count += 1
                    all_violations.append((i, constraint.description, violation))
                    
                    # Get current objective value estimate for adaptive scaling
                    # This ensures penalty is always significant relative to objective
                    current_obj_value = self.config.get('current_objective_value', None)
                    if current_obj_value is not None:
                        # Use actual objective value for precise scaling
                        objective_scale = max(1000.0, abs(current_obj_value) * 10.0)
                    else:
                        # Fallback to estimate
                        objective_scale = max(1000.0, abs(self.config.get('objective_scale_estimate', 1000.0)))
                    
                    # Ultra-aggressive penalty scaling:
                    # 1. For tiny violations: cubic penalty (violation^3)
                    # 2. For small violations: quartic penalty (violation^4)
                    # 3. For medium violations: exponential penalty (10^violation)
                    # 4. For large violations: double exponential (10^(violation*2))
                    
                    if violation < 0.1:
                        # Cubic penalty for tiny violations
                        scaled_penalty = objective_scale * (violation ** 3) * 10000.0
                    elif violation < 1.0:
                        # Quartic penalty for small violations
                        scaled_penalty = objective_scale * (violation ** 4) * 10000.0
                    elif violation < 10.0:
                        # Exponential penalty for medium violations
                        scaled_penalty = objective_scale * (10 ** violation) * 1000.0
                    else:
                        # Double exponential for large violations
                        scaled_penalty = objective_scale * (10 ** (min(violation * 2, 100))) * 1000.0
                        # Cap the exponent at 100 to avoid numerical overflow
                    
                    # Special handling for categorical features
                    if has_categorical:
                        # For categorical features, use even more extreme penalties
                        # to ensure discrete constraints are strictly enforced
                        scaled_penalty *= 10.0
                    
                    # Progressive multiplier for multiple violations
                    if violations_count > 1:
                        # Quadratic scaling with number of violations
                        scaled_penalty *= (violations_count ** 2)
                    
                    total_penalty += scaled_penalty
                    
                    if debug_output or violation > 1.0:
                        print(f"[CONSTRAINT {i+1}] Applied penalty: {scaled_penalty:.2f}")
                        print(f"[CONSTRAINT {i+1}] Total penalty so far: {total_penalty:.2f}")
                
            except Exception as e:
                print(f"[CONSTRAINT ERROR] Constraint {i+1} evaluation error: {str(e)}")
                traceback.print_exc()
                # Add a massive penalty for errors to discourage problematic solutions
                total_penalty += 500000.0
                violations_count += 1
                all_violations.append((i, f"ERROR: {str(e)}", 999.0))
                
        # Final detailed debugging output
        if total_penalty > 0:
            print(f"\n[CONSTRAINT SUMMARY] Final penalty: {total_penalty:.2f} for {violations_count} violations")
            print("[CONSTRAINT VIOLATIONS]")
            for i, desc, viol in all_violations:
                print(f"   - Constraint {i+1}: {desc} (violation: {viol:.6f})")
            
        # Store the penalty for debugging
        if not hasattr(self, '_last_penalty'):
            self._last_penalty = []
        self._last_penalty.append(total_penalty)
        
        # Store current objective estimate for future adaptive scaling
        if hasattr(self, 'last_objective_value'):
            self.config['current_objective_value'] = self.last_objective_value
            
        return total_penalty

    def _prepare_cobyla_constraints(self):
        """Prepare constraints specifically for COBYLA algorithm with enhanced categorical handling"""
        if not self.constraints:
            print("[COBYLA] No constraints defined")
            return []
            
        cobyla_constraints = []
        
        # Get feature types for better debugging
        feature_types = {}
        if hasattr(self, 'bounds_dict') and self.bounds_dict:
            for feature_name, bounds in self.bounds_dict.items():
                if bounds and hasattr(bounds, 'feature_type'):
                    feature_types[feature_name] = bounds.feature_type
        
        # Create feature index to name mapping for better debugging
        feature_index_to_name = {}
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                feature_index_to_name[i] = name
        
        print(f"\n[COBYLA] Setting up {len(self.constraints)} constraints")
        
        for i, constraint in enumerate(self.constraints):
            try:
                # Print detailed constraint information
                feature_indices_str = ", ".join(str(idx) for idx in constraint.feature_indices)
                feature_names = []
                has_categorical = False
                
                # Check if constraint involves categorical features
                for idx in constraint.feature_indices:
                    if idx < len(self.feature_names):
                        feature_name = self.feature_names[idx]
                        if feature_name in self.bounds_dict:
                            bounds = self.bounds_dict[feature_name]
                            feature_type = bounds.feature_type if bounds else "unknown"
                            feature_names.append(f"{feature_name}({feature_type})")
                            
                            if feature_type in ["categorical", "binary"]:
                                has_categorical = True
                        else:
                            feature_names.append(f"{feature_name}(unknown)")
                    else:
                        feature_names.append(f"unknown_{idx}")
                
                print(f"[COBYLA] Constraint {i+1}: {constraint.description}")
                print(f"[COBYLA] Features: {', '.join(feature_names)}")
                print(f"[COBYLA] Operator: {constraint.operator}, Bound: {constraint.bound}")
                if has_categorical:
                    print(f"[COBYLA] WARNING: This constraint includes categorical features!")
                
                # COBYLA expects constraint functions that return >= 0 when satisfied
                def make_constraint_func(constraint, constraint_id, has_categorical=False):
                    def constraint_func(x):
                        # Always apply categorical constraints first
                        # This is critical for consistent constraint evaluation
                        x_constrained = self._apply_categorical_constraints(x)
                        
                        # Debug tracking
                        if not hasattr(constraint_func, 'eval_count'):
                            constraint_func.eval_count = 0
                        constraint_func.eval_count += 1
                        
                        debug_output = constraint_func.eval_count <= 5
                        
                        # Calculate constraint value with detailed debugging
                        constraint_value = 0.0
                        feature_details = []
                        
                        for idx, coeff in zip(constraint.feature_indices, constraint.coefficients):
                            if idx < len(x_constrained):
                                feature_value = x_constrained[idx]
                                term_value = coeff * feature_value
                                constraint_value += term_value
                                
                                # Get feature name and type for debugging
                                feature_name = feature_index_to_name.get(idx, f"unknown_{idx}")
                                feature_type = "unknown"
                                if feature_name in self.bounds_dict:
                                    bounds = self.bounds_dict[feature_name]
                                    if bounds:
                                        feature_type = bounds.feature_type
                                
                                feature_details.append((idx, feature_name, feature_type, feature_value, coeff, term_value))
                            else:
                                print(f"[COBYLA ERROR] Constraint {constraint_id} references feature index {idx} which is out of bounds for input vector of length {len(x_constrained)}")
                                return -1.0  # Return constraint violation
                        
                        # Determine result based on operator
                        if constraint.operator == "<=":
                            result = constraint.bound - constraint_value  # bound - value >= 0
                            satisfied = result >= 0
                        elif constraint.operator == ">=":
                            result = constraint_value - constraint.bound  # value - bound >= 0
                            satisfied = result >= 0
                        elif constraint.operator == "==":
                            # For equality constraints, use a tolerance
                            # Adjust tolerance based on whether categorical features are involved
                            tolerance = 1e-8 if has_categorical else 1e-6
                            deviation = abs(constraint_value - constraint.bound)
                            result = tolerance - deviation
                            satisfied = result >= 0
                        else:
                            print(f"[COBYLA ERROR] Unknown operator: {constraint.operator}")
                            return -1.0
                        
                        # Print detailed debugging info
                        if debug_output or not satisfied:
                            print(f"\n[COBYLA] Evaluating constraint {constraint_id}: {constraint.description}")
                            print(f"[COBYLA] Feature values:")
                            for idx, name, type_name, value, coeff, term in feature_details:
                                print(f"   - {name}({idx}, {type_name}): {value:.6f} × {coeff:.6f} = {term:.6f}")
                            
                            print(f"[COBYLA] Total value: {constraint_value:.6f} {constraint.operator} {constraint.bound:.6f}")
                            print(f"[COBYLA] Result: {result:.6f} ({'SATISFIED' if satisfied else 'VIOLATED'})")
                        
                        return result
                    
                    # Initialize eval count
                    constraint_func.eval_count = 0
                    return constraint_func
                
                # Create constraint function with categorical awareness
                cobyla_constraints.append({
                    'type': 'ineq',
                    'fun': make_constraint_func(constraint, i+1, has_categorical)
                })
                
                # For equality constraints with non-categorical features, add a second constraint
                # This ensures the value is both <= bound+tolerance and >= bound-tolerance
                if constraint.operator == "==" and not has_categorical:
                    def make_eq_constraint_func(constraint, constraint_id):
                        def constraint_func(x):
                            # Apply categorical constraints first
                            x_constrained = self._apply_categorical_constraints(x)
                            
                            # Calculate constraint value
                            constraint_value = 0.0
                            for idx, coeff in zip(constraint.feature_indices, constraint.coefficients):
                                if idx < len(x_constrained):
                                    constraint_value += coeff * x_constrained[idx]
                                else:
                                    return -1.0  # Return constraint violation
                            
                            # This one checks that value >= bound - tolerance
                            result = constraint_value - (constraint.bound - 1e-6)
                            
                            # Debug output
                            if not hasattr(constraint_func, 'eval_count'):
                                constraint_func.eval_count = 0
                            constraint_func.eval_count += 1
                            
                            if constraint_func.eval_count <= 5 or result < 0:
                                print(f"[COBYLA] Constraint {constraint_id}b: {constraint_value:.6f} >= {constraint.bound - 1e-6:.6f} (result: {result:.6f})")
                            
                            return result
                        
                        # Initialize eval count
                        constraint_func.eval_count = 0
                        return constraint_func
                    
                    cobyla_constraints.append({
                        'type': 'ineq',
                        'fun': make_eq_constraint_func(constraint, f"{i+1}b")
                    })
                
                print(f"[COBYLA] Successfully added constraint {i+1}")
                
            except Exception as e:
                print(f"[COBYLA ERROR] Failed to create constraint {i+1}: {str(e)}")
                traceback.print_exc()
                
        print(f"[COBYLA] Total constraints created: {len(cobyla_constraints)}")
        return cobyla_constraints

    def _cobyla_optimization(self, objective, bounds, constraints=None, callback=None):
        """Enhanced COBYLA (Constrained Optimization BY Linear Approximation) implementation"""
        if not SCIPY_AVAILABLE:
            raise ValueError("COBYLA requires scipy")
        
        # Prepare COBYLA-specific constraints
        cobyla_constraints = self._prepare_cobyla_constraints()
        
        # Add bounds as constraints for COBYLA (more robust implementation)
        bounds_constraints = []
        for i, (min_val, max_val) in enumerate(bounds):
            if abs(min_val - max_val) > 1e-10:  # Not fixed (allow small numerical differences)
                # Lower bound constraint: x[i] - min_val >= 0
                bounds_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, min_val=min_val: x[i] - min_val
                })
                # Upper bound constraint: max_val - x[i] >= 0
                bounds_constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, max_val=max_val: max_val - x[i]
                })
            else:
                # For fixed variables, add equality constraint
                fixed_val = min_val
                bounds_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, fixed_val=fixed_val: 1e-6 - abs(x[i] - fixed_val)
                })
        
        # Combine all constraints
        all_constraints = cobyla_constraints + bounds_constraints
        
        # Generate smart initial point within bounds
        x0 = []
        for min_val, max_val in bounds:
            if min_val == max_val:
                x0.append(min_val)
            else:
                # Start closer to center of bounds for better convergence
                center = (min_val + max_val) / 2
                x0.append(center)
        x0 = np.array(x0)
        
        # Validate initial point against constraints
        for i, constraint in enumerate(all_constraints):
            try:
                if constraint['fun'](x0) < 0:
                    print(f"[WARNING] Initial point violates constraint {i}, adjusting...")
                    # Try to adjust initial point (simple heuristic)
                    for j in range(len(x0)):
                        if bounds[j][0] != bounds[j][1]:  # Not fixed
                            x0[j] = bounds[j][0] + 0.1 * (bounds[j][1] - bounds[j][0])
                    break
            except:
                continue
        
        # Enhanced COBYLA-specific options
        rhobeg = self.config.get('rhobeg', 1.0)
        rhoend = self.config.get('rhoend', 1e-6)
        
        # Adaptive trust region sizing based on problem scale
        bounds_range = np.array([max_val - min_val for min_val, max_val in bounds if min_val != max_val])
        if len(bounds_range) > 0:
            avg_range = np.mean(bounds_range)
            rhobeg = min(rhobeg, avg_range * 0.1)  # 10% of average range
            rhoend = min(rhoend, rhobeg * 1e-4)
        
        options = {
            'maxiter': self.config.get('max_iterations', 100),
            'rhobeg': rhobeg,
            'rhoend': rhoend,
            'disp': False
        }
        
        iteration_count = 0
        best_x = x0.copy()
        best_fun = float('inf')
        
        def callback_wrapper(x):
            nonlocal iteration_count, best_x, best_fun
            iteration_count += 1
            
            if callback:
                # Calculate objective value for display
                try:
                    obj_val = objective(x)
                    if obj_val < best_fun:
                        best_fun = obj_val
                        best_x = x.copy()
                    
                    # Check constraint satisfaction
                    constraints_satisfied = True
                    for constraint in all_constraints:
                        try:
                            if constraint['fun'](x) < -1e-6:
                                constraints_satisfied = False
                                break
                        except:
                            constraints_satisfied = False
                            break
                    
                    callback(iteration_count, obj_val, x, {
                        'algorithm': 'COBYLA',
                        'constraints_satisfied': constraints_satisfied,
                        'n_constraints': len(all_constraints),
                        'trust_region_radius': rhobeg * (rhoend/rhobeg)**(iteration_count/options['maxiter'])
                    })
                except Exception as e:
                    if self._debug_call_count <= 3:
                        print(f"[DEBUG] Callback error: {e}")
            
            return self.should_stop
        
        try:
            # Run COBYLA optimization with enhanced error handling
            result = minimize(
                objective,
                x0,
                method='COBYLA',
                constraints=all_constraints,
                callback=callback_wrapper,
                options=options
            )
            
            # Validate result
            if result.x is None or len(result.x) == 0:
                result.x = best_x
                result.fun = best_fun
                result.success = False
                result.message = "COBYLA returned invalid result, using best found"
            
            # Final constraint check
            constraints_satisfied = True
            constraint_violations = []
            for i, constraint in enumerate(all_constraints):
                try:
                    val = constraint['fun'](result.x)
                    if val < -1e-6:
                        constraints_satisfied = False
                        constraint_violations.append(f"Constraint {i}: {val:.6f}")
                except:
                    constraints_satisfied = False
                    constraint_violations.append(f"Constraint {i}: evaluation failed")
            
            return {
                'x': result.x,
                'fun': result.fun,
                'success': result.success and constraints_satisfied,
                'message': result.message,
                'nfev': result.nfev,
                'nit': result.nit if hasattr(result, 'nit') else iteration_count,
                'algorithm': 'COBYLA',
                'constraints_satisfied': constraints_satisfied,
                'constraint_violations': constraint_violations,
                'n_constraints': len(all_constraints),
                'rhobeg': rhobeg,
                'rhoend': rhoend
            }
            
        except Exception as e:
            error_msg = f"COBYLA optimization failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            
            # Return best found solution as fallback
            return {
                'x': best_x,
                'fun': best_fun,
                'success': False,
                'message': error_msg,
                'algorithm': 'COBYLA',
                'constraints_satisfied': False,
                'error': str(e)
            }
    
    def _prepare_scipy_constraints(self):
        """Prepare constraints for scipy optimization algorithms"""
        if not self.constraints or not SCIPY_AVAILABLE:
            return []
            
        scipy_constraints = []
        
        for constraint in self.constraints:
            try:
                # Create constraint matrix
                A = np.zeros(len(self.feature_names))
                for idx, coeff in zip(constraint.feature_indices, constraint.coefficients):
                    if idx < len(A):
                        A[idx] = coeff
                
                # Create scipy constraint
                if constraint.operator == "<=":
                    scipy_constraints.append(LinearConstraint(A, -np.inf, constraint.bound))
                elif constraint.operator == ">=":
                    scipy_constraints.append(LinearConstraint(A, constraint.bound, np.inf))
                elif constraint.operator == "==":
                    scipy_constraints.append(LinearConstraint(A, constraint.bound, constraint.bound))
                    
            except Exception as e:
                continue
                
        return scipy_constraints
        
    def _prepare_bounds(self):
        """Prepare bounds for optimization algorithms"""
        if not self.bounds_dict:
            # Create default bounds if none exist
            self._create_default_bounds()
            
        bounds = []
        for feature_name, bound_config in self.bounds_dict.items():
            if bound_config is None:
                # Handle None bound_config
                bounds.append((0.0, 1.0))  # Default bounds
            elif bound_config.is_fixed:
                # For fixed values, use very tight bounds
                bounds.append((bound_config.fixed_value, bound_config.fixed_value))
            else:
                bounds.append((bound_config.min_value, bound_config.max_value))
        return bounds
    
    def _prepare_bounds_and_space(self):
        """Prepare bounds and space configuration for different optimization algorithms
        
        CRITICAL FIX: Use self.feature_names as the authoritative source of feature order
        rather than relying on dictionary iteration order which is not guaranteed to be consistent.
        """
        if not self.bounds_dict:
            self._create_default_bounds()
            
        bounds = []
        space_config = []  # For scikit-optimize
        
        # CRITICAL FIX: Always use self.feature_names for consistent ordering
        if not self.feature_names:
            # Fallback if feature_names is not available
            print("[WARNING] No feature_names available, using bounds_dict keys for ordering")
            ordered_features = list(self.bounds_dict.keys())
        else:
            # Use feature_names as the authoritative source of order
            ordered_features = self.feature_names
            
        print(f"[DEBUG] Preparing bounds for {len(ordered_features)} features in strict order")
        print(f"[DEBUG] Feature names order: {ordered_features}")
        print(f"[DEBUG] Bounds dict keys: {list(self.bounds_dict.keys())}")
        
        # Process features in the exact order specified by ordered_features
        for i, feature_name in enumerate(ordered_features):
            try:
                # Skip features not in bounds_dict (should not happen in normal operation)
                if feature_name not in self.bounds_dict:
                    print(f"[WARNING] Feature {feature_name} not found in bounds_dict, using default bounds")
                    bounds.append((0.0, 1.0))
                    if SKOPT_AVAILABLE:
                        space_config.append(Real(0.0, 1.0, name=feature_name))
                    continue
                    
                # Get bound configuration for this feature
                bound_config = self.bounds_dict[feature_name]
                
                if bound_config is None:
                    bounds.append((0.0, 1.0))
                    if SKOPT_AVAILABLE:
                        space_config.append(Real(0.0, 1.0, name=feature_name))
                elif bound_config.is_fixed:
                    fixed_val = bound_config.fixed_value
                    bounds.append((fixed_val, fixed_val))
                    print(f"[DEBUG] Fixed bound {i}: {feature_name} = [{fixed_val}, {fixed_val}]")
                    if SKOPT_AVAILABLE:
                        # For fixed values, create a very small range to satisfy scikit-optimize
                        epsilon = max(1e-8, abs(fixed_val) * 1e-10) if fixed_val != 0 else 1e-8
                        try:
                            space_config.append(Real(fixed_val - epsilon, fixed_val + epsilon, name=feature_name))
                        except ValueError as e:
                            print(f"[WARNING] Failed to create Real space for fixed feature {feature_name}: {e}")
                            # Fallback: skip this feature for scikit-optimize algorithms
                            pass
                else:
                    min_val, max_val = bound_config.min_value, bound_config.max_value
                    
                    # Ensure min < max
                    if min_val >= max_val:
                        print(f"[WARNING] Invalid bounds for {feature_name}: min={min_val}, max={max_val}")
                        # Swap or adjust bounds
                        if min_val == max_val:
                            # Treat as fixed value
                            bounds.append((min_val, min_val))
                            print(f"[DEBUG] Treated as fixed bound {i}: {feature_name} = [{min_val}, {min_val}]")
                            if SKOPT_AVAILABLE:
                                epsilon = max(1e-8, abs(min_val) * 1e-10) if min_val != 0 else 1e-8
                                try:
                                    space_config.append(Real(min_val - epsilon, min_val + epsilon, name=feature_name))
                                except ValueError:
                                    pass
                        else:
                            # Swap bounds
                            min_val, max_val = max_val, min_val
                            bounds.append((min_val, max_val))
                            print(f"[DEBUG] Swapped bound {i}: {feature_name} = [{min_val}, {max_val}]")
                            if SKOPT_AVAILABLE:
                                space_config.append(Real(min_val, max_val, name=feature_name))
                    else:
                        bounds.append((min_val, max_val))
                        print(f"[DEBUG] Normal bound {i}: {feature_name} = [{min_val}, {max_val}] (type: {bound_config.feature_type})")
                        
                        if SKOPT_AVAILABLE:
                            if bound_config.feature_type == "categorical":
                                space_config.append(Categorical(bound_config.categorical_values, name=feature_name))
                            elif bound_config.feature_type == "integer":
                                space_config.append(Integer(int(min_val), int(max_val), name=feature_name))
                            else:
                                try:
                                    space_config.append(Real(min_val, max_val, name=feature_name))
                                except ValueError as e:
                                    print(f"[WARNING] Failed to create Real space for {feature_name}: {e}")
                                    # Create default bounds
                                    space_config.append(Real(0.0, 1.0, name=feature_name))
                                    
            except Exception as e:
                print(f"[ERROR] Failed to process bounds for feature {feature_name}: {e}")
                # Add default bounds as fallback
                bounds.append((0.0, 1.0))
                if SKOPT_AVAILABLE:
                    space_config.append(Real(0.0, 1.0, name=feature_name))
        
        # Verify bounds length matches feature count
        if len(bounds) != len(ordered_features):
            print(f"[ERROR] Bounds length mismatch: {len(bounds)} bounds for {len(ordered_features)} features")
            
        # Print comprehensive bounds summary
        print(f"[DEBUG] ===== FINAL BOUNDS SUMMARY =====")
        print(f"[DEBUG] Total bounds: {len(bounds)}")
        for i, ((min_val, max_val), feature_name) in enumerate(zip(bounds, ordered_features)):
            print(f"[DEBUG] [{i:2d}] {feature_name:20s} = [{min_val:10.6f}, {max_val:10.6f}]")
        print(f"[DEBUG] =================================")
            
        return bounds, space_config
        
    def _create_default_bounds(self):
        """Create default bounds for all features"""
        # Get feature names from bounds_dict keys or create generic names
        if not self.bounds_dict:
            # Create default feature names if none exist
            n_features = self._get_n_features()
            feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            feature_names = list(self.bounds_dict.keys())
            
        # Create default bounds with realistic ranges
        for feature_name in feature_names:
            if feature_name not in self.bounds_dict:
                # Use realistic bounds based on feature name
                feature_upper = feature_name.upper()
                if "ENERGY" in feature_upper:
                    min_val, max_val = 0.0, 100.0
                elif "AGE" in feature_upper:
                    min_val, max_val = 0.0, 100.0
                elif "OXYGEN" in feature_upper:
                    min_val, max_val = 80.0, 100.0
                elif any(x in feature_upper for x in ["RADII", "RADIUS"]):
                    min_val, max_val = 0.5, 3.0
                elif any(x in feature_upper for x in ["WEIGHT", "MASS"]):
                    min_val, max_val = 1.0, 300.0
                elif "CHARGE" in feature_upper:
                    min_val, max_val = 1.0, 50.0
                elif "HEAT" in feature_upper:
                    min_val, max_val = 100.0, 5000.0
                elif "LATTICE" in feature_upper:
                    min_val, max_val = 200.0, 800.0
                else:
                    min_val, max_val = 0.0, 100.0
                    
                self.bounds_dict[feature_name] = FeatureBounds(
                    min_value=min_val,
                    max_value=max_val,
                    is_fixed=False
                )
                
    def _get_n_features(self):
        """Get number of features from model"""
        if hasattr(self.model, 'n_features_in_'):
            return self.model.n_features_in_
        elif hasattr(self.model, 'steps'):
            # For pipeline, check the last step
            final_estimator = self.model.steps[-1][1]
            if hasattr(final_estimator, 'n_features_in_'):
                return final_estimator.n_features_in_
        return 10  # Default fallback
    
    def _grid_search(self, objective, bounds, feature_names):
        """Optimized grid search with reduced UI updates"""
        n_points = self.config.get('n_points_per_dim', 8)  # Reduced from 10 to 8 for speed
        
        # Generate grid points
        grid_ranges = []
        for min_val, max_val in bounds:
            if min_val == max_val:  # Fixed value
                grid_ranges.append([min_val])
            else:
                grid_ranges.append(np.linspace(min_val, max_val, n_points))
        
        best_val = float('inf')
        best_x = None
        total_combinations = np.prod([len(gr) for gr in grid_ranges])
        
        iteration = 0
        update_frequency = max(1, total_combinations // 50)  # Update UI only 50 times max
        
        # Generate all combinations
        for combination in itertools.product(*grid_ranges):
            if self.should_stop:
                break
                
            val = objective(list(combination))
            
            if val < best_val:
                best_val = val
                best_x = list(combination)
                
                display_val = -val if self.target_direction == "maximize" else val
                self.best_value = display_val
                self.best_params = list(best_x)
            
            self.history.append((list(combination), val))
            
            iteration += 1
            
            # Update UI only periodically to improve performance
            if iteration % update_frequency == 0 or iteration == total_combinations:
                progress = int(iteration / total_combinations * 90)
                self.progress_updated.emit(progress)
                
                if self.best_value is not None:
                    self.iteration_completed.emit(iteration, self.best_value, self.best_params, {})
                
                # Small delay to allow UI processing
                self.msleep(1)
        
        # Ensure we have a valid result
        if best_x is None:
            # Create fallback result
            best_x = []
            for min_val, max_val in bounds:
                if min_val == max_val:
                    best_x.append(min_val)
                else:
                    best_x.append((min_val + max_val) / 2)  # Use midpoint
            best_val = objective(best_x)
            
        return {'x': best_x, 'fun': best_val}
        
    def _random_search(self, objective, bounds, feature_names):
        """Optimized random search with batch processing"""
        n_iterations = self.config.get('n_iterations', 500)  # Reduced from 1000 to 500
        
        best_val = float('inf')
        best_x = None
        update_frequency = max(1, n_iterations // 50)  # Update UI only 50 times max
        
        for i in range(n_iterations):
            if self.should_stop:
                break
                
            # Random sample within bounds
            x = []
            for min_val, max_val in bounds:
                if min_val == max_val:  # Fixed value
                    x.append(min_val)
                else:
                    x.append(np.random.uniform(min_val, max_val))
            
            val = objective(x)
            
            if val < best_val:
                best_val = val
                best_x = x
                
                display_val = -val if self.target_direction == "maximize" else val
                self.best_value = display_val
                self.best_params = list(best_x)
            
            self.history.append((list(x), val))
            
            # Update UI only periodically
            if (i + 1) % update_frequency == 0 or i == n_iterations - 1:
                progress = int((i + 1) / n_iterations * 90)
                self.progress_updated.emit(progress)
                
                if self.best_value is not None:
                    self.iteration_completed.emit(i + 1, self.best_value, self.best_params, {})
                
                # Small delay to allow UI processing
                self.msleep(1)
        
        # Ensure we have a valid result
        if best_x is None:
            # Create fallback result
            best_x = []
            for min_val, max_val in bounds:
                if min_val == max_val:
                    best_x.append(min_val)
                else:
                    best_x.append(np.random.uniform(min_val, max_val))
            best_val = objective(best_x)
            
        return {'x': best_x, 'fun': best_val}
        
    def _differential_evolution(self, objective, bounds, constraints=None, callback=None):
        """Enhanced differential evolution optimization with constraint support"""
        maxiter = self.config.get('maxiter', 100)
        popsize = self.config.get('popsize', 15)
        
        def callback_wrapper(xk, convergence):
            if self.should_stop:
                return True
                
            val = objective(xk)
            display_val = -val if self.target_direction == "maximize" else val
            
            if self.best_value is None or (self.target_direction == "maximize" and display_val > self.best_value) or \
               (self.target_direction == "minimize" and display_val < self.best_value):
                self.best_value = display_val
                self.best_params = list(xk)
                
            self.iteration_completed.emit(len(self.history), display_val, 
                                        self.best_params, {
                                            'population_fitness': val,
                                            'generation': len(self.history)
                                        })
            
            self.history.append((self.best_params, val))
            
            progress = min(90, int(len(self.history) / (maxiter * popsize) * 90))
            self.progress_updated.emit(progress)
            
            return False
        
        # Use constraints if available
        result = differential_evolution(
            objective, bounds, maxiter=maxiter, popsize=popsize,
            callback=callback_wrapper, seed=42,
            constraints=constraints if constraints else ()
        )
        
        return result
        
    def _basin_hopping(self, objective, bounds, constraints=None, callback=None):
        """Enhanced basin hopping optimization with constraint support"""
        niter = self.config.get('niter', 100)
        
        # Initial guess
        x0 = []
        for min_val, max_val in bounds:
            if min_val == max_val:  # Fixed value
                x0.append(min_val)
            else:
                x0.append(np.random.uniform(min_val, max_val))
        
        def callback_wrapper(x, f, accept):
            if self.should_stop:
                return True
                
            display_val = -f if self.target_direction == "maximize" else f
            
            if self.best_value is None or (self.target_direction == "maximize" and display_val > self.best_value) or \
               (self.target_direction == "minimize" and display_val < self.best_value):
                self.best_value = display_val
                self.best_params = list(x)
                
            self.iteration_completed.emit(len(self.history), display_val, 
                                        self.best_params, {
                                            'population_fitness': f,
                                            'generation': len(self.history)
                                        })
            
            self.history.append((self.best_params, f))
            
            progress = min(90, int(len(self.history) / niter * 90))
            self.progress_updated.emit(progress)
            
            return False
        
        # Define bounds for scipy
        from scipy.optimize import Bounds
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        bounds_obj = Bounds(lower_bounds, upper_bounds)
        
        # Prepare minimizer kwargs with constraints
        minimizer_kwargs = {'bounds': bounds_obj, 'method': 'L-BFGS-B'}
        if constraints:
            minimizer_kwargs['constraints'] = constraints
        
        result = basinhopping(
            objective, x0, niter=niter, callback=callback_wrapper,
            minimizer_kwargs=minimizer_kwargs
        )
        
        return result
        
    def _bayesian_optimization(self, objective, space_config, callback=None):
        """Enhanced Bayesian optimization using scikit-optimize with categorical support"""
        n_calls = self.config.get('n_calls', 100)
        
        # Use provided space configuration or create default
        if isinstance(space_config, list) and len(space_config) > 0:
            dimensions = space_config
        else:
            # Fallback to bounds-based approach
            bounds = space_config if isinstance(space_config, list) else []
            dimensions = []
            for min_val, max_val in bounds:
                if min_val == max_val:  # Fixed value
                    dimensions.append(Real(low=min_val, high=min_val + 1e-8))
                else:
                    dimensions.append(Real(low=min_val, high=max_val))
        
        iteration_count = [0]
        
        def callback_wrapper(result):
            if self.should_stop:
                return True
                
            x = result.x_iters[-1]
            val = result.func_vals[-1]
            
            display_val = -val if self.target_direction == "maximize" else val
            
            if self.best_value is None or (self.target_direction == "maximize" and display_val > self.best_value) or \
               (self.target_direction == "minimize" and display_val < self.best_value):
                self.best_value = display_val
                self.best_params = list(x)
                
            self.iteration_completed.emit(iteration_count[0], display_val, 
                                        self.best_params, {
                                            'population_fitness': val,
                                            'generation': iteration_count[0]
                                        })
            
            self.history.append((self.best_params, val))
            
            progress = min(90, int(iteration_count[0] / n_calls * 90))
            self.progress_updated.emit(progress)
            
            iteration_count[0] += 1
            return False
        
        result = gp_minimize(
            objective, dimensions, n_calls=n_calls,
            callback=callback_wrapper, random_state=42
        )
        
        return result

    def _convert_to_original_categories(self, x_numeric):
        """Convert numeric values from optimizer back to original categorical values for model prediction
        
        Args:
            x_numeric: Numeric vector from optimizer with constrained categorical values
            
        Returns:
            A list with mixed types - numbers for continuous features and original values for categorical features
        """
        # Make a copy that we'll modify
        x_mixed = list(x_numeric)  # Convert to list to support mixed types
        
        # Only proceed if we have categorical mappings
        if not self.categorical_mappings:
            return x_mixed
            
        # For each feature, check if it's categorical and convert if needed
        for i, feature_name in enumerate(self.feature_names):
            if i >= len(x_mixed):
                continue
                
            if feature_name in self.categorical_mappings:
                # This is a categorical feature - convert numeric value back to original
                numeric_value = int(round(x_mixed[i]))  # Round to nearest integer
                
                # Get mapping for this feature
                mapping = self.categorical_mappings[feature_name]['num_to_orig']
                
                # Convert back to original value if possible
                if numeric_value in mapping:
                    x_mixed[i] = mapping[numeric_value]
                    
                    # Debug output for first few conversions
                    if not hasattr(self, '_debug_conversion_count'):
                        self._debug_conversion_count = 0
                    
                    self._debug_conversion_count += 1
                    if self._debug_conversion_count <= 10:
                        print(f"[CATEGORICAL] Converted {feature_name} from {numeric_value} to {x_mixed[i]}")
                else:
                    # If the numeric value is out of range, use the closest valid value
                    valid_values = list(mapping.keys())
                    if valid_values:
                        closest_idx = min(range(len(valid_values)), key=lambda j: abs(valid_values[j] - numeric_value))
                        closest_value = valid_values[closest_idx]
                        x_mixed[i] = mapping[closest_value]
                        print(f"[WARNING] Invalid categorical value {numeric_value} for {feature_name}, using closest: {x_mixed[i]}")
        
        return x_mixed

class TargetOptimizationModule(QWidget):
    """Target Optimization Module"""
    
    optimization_completed = pyqtSignal(dict)
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_names = None
        self.feature_types = {}
        self.feature_bounds = {}
        self.constraints = []
        self.optimization_results = None
        self.worker = None
        self.training_data = None  # Added to store training data
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎯 Target Optimization Module")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { color: #2196F3; margin: 10px; }")
        layout.addWidget(title)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - Configuration
        left_panel = self.create_configuration_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self.create_results_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)
        
        # Status and progress
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #666666; margin: 5px; }")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
    def create_configuration_panel(self):
        """Create configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Algorithm Configuration
        algo_group = QGroupBox("⚙️ Algorithm Configuration")
        algo_layout = QVBoxLayout(algo_group)
        
        # Algorithm selection
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        algorithms = ["Genetic Algorithm", "Particle Swarm Optimization", "Grid Search", "Random Search"]
        if SCIPY_AVAILABLE:
            algorithms.extend(["Differential Evolution", "Basin Hopping", "COBYLA"])
        if SKOPT_AVAILABLE:
            algorithms.append("Bayesian Optimization")
        self.algorithm_combo.addItems(algorithms)
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algo_row.addWidget(self.algorithm_combo)
        algo_layout.addLayout(algo_row)
        
        # Target direction
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems(["minimize", "maximize"])
        target_row.addWidget(self.target_combo)
        algo_layout.addLayout(target_row)
        
        # Number of iterations/generations
        iter_row = QHBoxLayout()
        self.iter_label = QLabel("Iterations:")
        iter_row.addWidget(self.iter_label)
        self.n_iterations_spin = QSpinBox()
        self.n_iterations_spin.setRange(10, 100000)
        self.n_iterations_spin.setValue(100)
        iter_row.addWidget(self.n_iterations_spin)
        algo_layout.addLayout(iter_row)
        
        # Population size (for GA/PSO)
        pop_row = QHBoxLayout()
        self.pop_label = QLabel("Population Size:")
        pop_row.addWidget(self.pop_label)
        self.population_spin = QSpinBox()
        self.population_spin.setRange(10, 10000)
        self.population_spin.setValue(50)
        pop_row.addWidget(self.population_spin)
        algo_layout.addLayout(pop_row)
        
        # Grid points (for grid search)
        grid_row = QHBoxLayout()
        self.grid_label = QLabel("Grid Points/Dim:")
        grid_row.addWidget(self.grid_label)
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(3, 100)
        self.n_points_spin.setValue(6)
        grid_row.addWidget(self.n_points_spin)
        algo_layout.addLayout(grid_row)
        
        # Mutation rate (for GA)
        mut_row = QHBoxLayout()
        self.mut_label = QLabel("Mutation Rate:")
        mut_row.addWidget(self.mut_label)
        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setRange(0.01, 1.0)
        self.mutation_rate_spin.setValue(0.1)
        self.mutation_rate_spin.setSingleStep(0.01)
        mut_row.addWidget(self.mutation_rate_spin)
        algo_layout.addLayout(mut_row)
        
        # Crossover rate (for GA)
        cross_row = QHBoxLayout()
        self.cross_label = QLabel("Crossover Rate:")
        cross_row.addWidget(self.cross_label)
        self.crossover_rate_spin = QDoubleSpinBox()
        self.crossover_rate_spin.setRange(0.1, 1.0)
        self.crossover_rate_spin.setValue(0.8)
        self.crossover_rate_spin.setSingleStep(0.01)
        cross_row.addWidget(self.crossover_rate_spin)
        algo_layout.addLayout(cross_row)
        
        # Inertia weight (for PSO)
        inertia_row = QHBoxLayout()
        self.inertia_label = QLabel("Inertia Weight:")
        inertia_row.addWidget(self.inertia_label)
        self.inertia_weight_spin = QDoubleSpinBox()
        self.inertia_weight_spin.setRange(0.1, 1.5)
        self.inertia_weight_spin.setValue(0.9)
        self.inertia_weight_spin.setSingleStep(0.01)
        inertia_row.addWidget(self.inertia_weight_spin)
        algo_layout.addLayout(inertia_row)
        
        # Trust region radius (for COBYLA)
        trust_row = QHBoxLayout()
        self.trust_label = QLabel("Trust Region Radius:")
        trust_row.addWidget(self.trust_label)
        self.trust_radius_spin = QDoubleSpinBox()
        self.trust_radius_spin.setRange(0.001, 1000.0)
        self.trust_radius_spin.setValue(1.0)
        self.trust_radius_spin.setSingleStep(0.1)
        trust_row.addWidget(self.trust_radius_spin)
        algo_layout.addLayout(trust_row)
        
        # Real-time visualization toggle
        viz_row = QHBoxLayout()
        self.realtime_viz_checkbox = QCheckBox("Real-time Visualization")
        self.realtime_viz_checkbox.setChecked(True)
        self.realtime_viz_checkbox.setToolTip("Update plots during optimization (may slow down)")
        viz_row.addWidget(self.realtime_viz_checkbox)
        algo_layout.addLayout(viz_row)
        
        # Performance visualization
        perf_row = QHBoxLayout()
        self.performance_viz_checkbox = QCheckBox("Performance Metrics")
        self.performance_viz_checkbox.setChecked(True)
        self.performance_viz_checkbox.setToolTip("Show algorithm performance metrics")
        perf_row.addWidget(self.performance_viz_checkbox)
        algo_layout.addLayout(perf_row)
        
        layout.addWidget(algo_group)
        
        # Feature bounds configuration
        bounds_group = QGroupBox("Feature Bounds")
        bounds_layout = QVBoxLayout(bounds_group)
        
        self.bounds_table = QTableWidget()
        self.bounds_table.setColumnCount(4)
        self.bounds_table.setHorizontalHeaderLabels(["Feature", "Min", "Max", "Fixed"])
        bounds_layout.addWidget(self.bounds_table)
        
        layout.addWidget(bounds_group)
        
        # Constraints configuration
        constraints_group = QGroupBox("📏 Constraints")
        constraints_layout = QVBoxLayout(constraints_group)
        
        # Constraints table
        self.constraints_table = QTableWidget()
        self.constraints_table.setColumnCount(4)
        self.constraints_table.setHorizontalHeaderLabels(["Features", "Operator", "Value", "Description"])
        self.constraints_table.setMaximumHeight(150)
        constraints_layout.addWidget(self.constraints_table)
        
        # Add constraint button
        add_constraint_btn = QPushButton("Add Constraint")
        add_constraint_btn.clicked.connect(self.add_constraint)
        constraints_layout.addWidget(add_constraint_btn)
        
        layout.addWidget(constraints_group)
        
        # Set initial visibility based on default algorithm
        self._on_algorithm_changed(self.algorithm_combo.currentText())
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Optimization")
        self.start_button.clicked.connect(self.start_optimization)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
        return panel
    
    def _on_algorithm_changed(self, algorithm):
        """Update UI based on selected algorithm"""
        # Hide all algorithm-specific controls first
        self.pop_label.setVisible(False)
        self.population_spin.setVisible(False)
        self.grid_label.setVisible(False)
        self.n_points_spin.setVisible(False)
        self.mut_label.setVisible(False)
        self.mutation_rate_spin.setVisible(False)
        self.cross_label.setVisible(False)
        self.crossover_rate_spin.setVisible(False)
        self.inertia_label.setVisible(False)
        self.inertia_weight_spin.setVisible(False)
        self.trust_label.setVisible(False)
        self.trust_radius_spin.setVisible(False)
        
        # Update labels and show relevant controls
        if algorithm == "Genetic Algorithm":
            self.iter_label.setText("Generations:")
            self.pop_label.setVisible(True)
            self.population_spin.setVisible(True)
            self.mut_label.setVisible(True)
            self.mutation_rate_spin.setVisible(True)
            self.cross_label.setVisible(True)
            self.crossover_rate_spin.setVisible(True)
        elif algorithm == "Particle Swarm Optimization":
            self.iter_label.setText("Iterations:")
            self.pop_label.setText("Particles:")
            self.pop_label.setVisible(True)
            self.population_spin.setVisible(True)
            self.inertia_label.setVisible(True)
            self.inertia_weight_spin.setVisible(True)
        elif algorithm == "Grid Search":
            self.iter_label.setText("Total Points:")
            self.grid_label.setVisible(True)
            self.n_points_spin.setVisible(True)
        elif algorithm == "COBYLA":
            self.iter_label.setText("Max Iterations:")
            self.trust_label.setVisible(True)
            self.trust_radius_spin.setVisible(True)
        else:  # Random Search, Differential Evolution, Basin Hopping, Bayesian Optimization
            self.iter_label.setText("Iterations:")
        
    def create_results_panel(self):
        """Create results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Optimization results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Export button
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_results)
        results_layout.addWidget(export_button)
        
        self.results_tabs.addTab(results_tab, "Results")
        
        # Visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Create matplotlib figure and toolbar
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, viz_tab)
        
        # Add toolbar and canvas
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        self.results_tabs.addTab(viz_tab, "📊 Interactive Visualization")
        
        layout.addWidget(self.results_tabs)
        
        return panel
        
    def set_model(self, trained_model, feature_names=None, feature_info=None, training_data=None):
        """Enhanced model setting with feature type detection, improved feature name extraction,
        and dynamic feature bounds calculation from training data
        
        Args:
            trained_model: The trained ML model
            feature_names: Optional list of feature names
            feature_info: Optional dictionary with feature metadata
            training_data: Optional training data (DataFrame or numpy array) to calculate feature bounds
        """
        self.model = trained_model
        self.training_data = training_data  # Store training data for reference
        
        # **CRITICAL FIX**: Initialize storage for original categorical values
        if not hasattr(self, 'original_categorical_values'):
            self.original_categorical_values = {}
        
        # **CRITICAL FIX**: Prioritize model's actual feature names over provided ones
        model_feature_names = None
        
        # Try to get feature names directly from the model first
        if hasattr(self.model, 'feature_names_in_'):
            model_feature_names = list(self.model.feature_names_in_)
            print(f"[DEBUG] Got feature names from model.feature_names_in_: {model_feature_names[:3]}...")
        elif hasattr(self.model, 'named_steps'):
            # For Pipeline, try to get from the first step (usually preprocessor)
            try:
                first_step_name, first_step = list(self.model.named_steps.items())[0]
                if hasattr(first_step, 'feature_names_in_'):
                    model_feature_names = list(first_step.feature_names_in_)
                    print(f"[DEBUG] Got feature names from {first_step_name}.feature_names_in_: {model_feature_names[:3]}...")
                elif hasattr(first_step, 'get_feature_names_out'):
                    # Try get_feature_names_out for transformers
                    try:
                        model_feature_names = list(first_step.get_feature_names_out())
                        print(f"[DEBUG] Got feature names from {first_step_name}.get_feature_names_out(): {model_feature_names[:3]}...")
                    except:
                        pass
            except Exception as e:
                print(f"[WARNING] Could not extract feature names from Pipeline: {e}")
        
        # Use feature names in order of priority:
        # 1. Model's actual feature names (most reliable)
        # 2. Provided feature names
        # 3. Fallback to generic names
        if model_feature_names:
            self.feature_names = model_feature_names
            print(f"[INFO] Using model's feature names ({len(self.feature_names)} features)")
        elif feature_names:
            self.feature_names = list(feature_names)
            print(f"[INFO] Using provided feature names ({len(self.feature_names)} features)")
            # **WARNING**: Check if provided names match model expectations
            if hasattr(self.model, 'feature_names_in_'):
                expected_names = list(self.model.feature_names_in_)
                if set(self.feature_names) != set(expected_names):
                    print(f"[WARNING] Provided feature names may not match model expectations!")
                    print(f"[WARNING] Provided: {self.feature_names[:3]}...")
                    print(f"[WARNING] Expected: {expected_names[:3]}...")
        else:
            # Fallback: use generic names
            self.feature_names = [f"feature_{i}" for i in range(self._get_n_features())]
            print(f"[WARNING] Using generic feature names ({len(self.feature_names)} features)")
        
        # **CRITICAL FIX**: Detect categorical features from training data
        if training_data is not None and isinstance(training_data, pd.DataFrame):
            print(f"[INFO] Analyzing training data for categorical features")
            for feature_name in self.feature_names:
                if feature_name in training_data.columns:
                    col = training_data[feature_name]
                    
                    # Case 1: Non-numeric data (object or category dtype)
                    if col.dtype == 'object' or col.dtype.name == 'category':
                        original_values = col.unique().tolist()
                        print(f"[CATEGORICAL] Detected non-numeric feature {feature_name} with values: {original_values}")
                        self.original_categorical_values[feature_name] = original_values
                    
                    # Case 2: Numeric data with few unique values (likely categorical)
                    elif len(col.unique()) <= 10:  # Threshold for considering numeric as categorical
                        unique_values = sorted(col.unique().tolist())
                        # Check if values are integers or close to integers
                        if all(abs(val - round(val)) < 1e-6 for val in unique_values):
                            print(f"[CATEGORICAL] Detected numeric categorical feature {feature_name} with values: {unique_values}")
                            self.original_categorical_values[feature_name] = unique_values
        
        # Detect feature types from feature names and info
        self._detect_feature_types(feature_info)
        
        if self.model is not None and self.feature_names is not None:
            # Setup feature bounds using training data if available
            self.setup_feature_bounds(training_data)
            self.start_button.setEnabled(True)
            self.status_label.setText("Model loaded from training session. Ready for optimization.")
            
            # **VALIDATION**: Print final feature setup for debugging
            print(f"[INFO] Model setup complete:")
            print(f"[INFO] - Model type: {type(self.model)}")
            print(f"[INFO] - Feature count: {len(self.feature_names)}")
            print(f"[INFO] - First few features: {self.feature_names[:5]}")
            
            # **CRITICAL FIX**: Print categorical feature information
            categorical_features = [f for f in self.original_categorical_values.keys()]
            if categorical_features:
                print(f"[INFO] - Detected {len(categorical_features)} categorical features: {categorical_features}")
                for feature in categorical_features:
                    print(f"[INFO]   * {feature}: {self.original_categorical_values[feature]}")
            
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = list(self.model.feature_names_in_)
                if set(self.feature_names) == set(expected_features):
                    print(f"[INFO] ✅ Feature names match model expectations")
                else:
                    print(f"[WARNING] ⚠️  Feature names may not match model expectations")
                    print(f"[WARNING] Expected: {expected_features[:3]}...")
                    print(f"[WARNING] Using: {self.feature_names[:3]}...")
    
    def _detect_feature_types(self, feature_info=None):
        """Detect feature types from names and additional info with enhanced One-Hot detection"""
        self.feature_types = {}
        
        # CRITICAL FIX: Detect One-Hot encoded features by pattern analysis
        one_hot_groups = self._detect_one_hot_groups()
        
        for feature_name in self.feature_names:
            # Default to continuous - this is the most common case
            feature_type = "continuous"
            categorical_values = None
            
            # ENHANCED: Check if this is part of a One-Hot encoded group
            if any(feature_name in group for group in one_hot_groups.values()):
                feature_type = "binary"
                categorical_values = [0, 1]
                print(f"[DETECT] One-Hot encoded feature detected: {feature_name}")
            else:
                # Original detection logic for other patterns
                feature_upper = feature_name.upper()
                
                # Only mark as binary if it's clearly a binary indicator
                if (feature_upper.endswith('_YES') or feature_upper.endswith('_NO') or
                    feature_upper.startswith('IS_') or feature_upper.startswith('HAS_') or
                    'BINARY' in feature_upper):
                    feature_type = "binary"
                    categorical_values = [0, 1]
                
                # Detect categorical features from naming patterns
                elif any(pattern in feature_upper for pattern in ['_CATEGORY', '_CLASS', '_TYPE']):
                    feature_type = "categorical"
                    # Default categorical values, can be overridden
                    categorical_values = [0, 1, 2]
            
            # Use provided feature info if available (this takes precedence)
            if feature_info and feature_name in feature_info:
                info = feature_info[feature_name]
                if 'type' in info:
                    feature_type = info['type']
                if 'values' in info:
                    categorical_values = info['values']
            
            self.feature_types[feature_name] = {
                'type': feature_type,
                'categorical_values': categorical_values
            }
        
        # Debug output
        print(f"[DETECT] Feature types detected:")
        for fname, finfo in self.feature_types.items():
            print(f"  {fname}: {finfo['type']}")
    
    def _detect_one_hot_groups(self):
        """Detect One-Hot encoded feature groups by name patterns"""
        one_hot_groups = {}
        
        for feature_name in self.feature_names:
            # Check for underscore pattern (common in One-Hot encoding)
            if '_' in feature_name:
                # Split by last underscore to get base name and value
                parts = feature_name.rsplit('_', 1)
                if len(parts) == 2:
                    base_name, value_part = parts
                    
                    # Only consider it One-Hot if the value part looks categorical
                    # (not numeric like _1, _2, _3 which might be feature indices)
                    if (value_part.isalpha() or  # Letters only (like Process_LJ, Process_RC)
                        len(value_part) <= 3 and not value_part.isdigit()):  # Short non-numeric
                        
                        if base_name not in one_hot_groups:
                            one_hot_groups[base_name] = []
                        one_hot_groups[base_name].append(feature_name)
        
        # Only keep groups with multiple features (actual One-Hot encoding)
        one_hot_groups = {k: v for k, v in one_hot_groups.items() if len(v) > 1}
        
        if one_hot_groups:
            print(f"[DETECT] One-Hot groups found:")
            for base_name, features in one_hot_groups.items():
                print(f"  {base_name}: {features}")
        
        return one_hot_groups
            
    def _get_n_features(self):
        """Get number of features from model"""
        if hasattr(self.model, 'n_features_in_'):
            return self.model.n_features_in_
        elif hasattr(self.model, 'steps'):
            # For pipeline, check the last step
            final_estimator = self.model.steps[-1][1]
            if hasattr(final_estimator, 'n_features_in_'):
                return final_estimator.n_features_in_
        return 10  # Default fallback
        
    def setup_feature_bounds(self, training_data=None):
        """Enhanced setup feature bounds table with dynamic range calculation from training data"""
        if not self.feature_names:
            return
            
        try:
            # Update table headers to include type
            self.bounds_table.setColumnCount(5)
            self.bounds_table.setHorizontalHeaderLabels(["Feature", "Min", "Max", "Fixed", "Type"])
            self.bounds_table.setRowCount(len(self.feature_names))
            
            # Calculate feature bounds from training data if available
            feature_ranges = {}
            
            if training_data is not None:
                print(f"[INFO] Calculating feature bounds from training data")
                
                # Convert to DataFrame if numpy array
                if isinstance(training_data, np.ndarray):
                    if len(self.feature_names) == training_data.shape[1]:
                        training_data = pd.DataFrame(training_data, columns=self.feature_names)
                    else:
                        print(f"[WARNING] Training data shape {training_data.shape} doesn't match feature count {len(self.feature_names)}")
                        training_data = None
                
                # Extract min/max values for each feature
                if isinstance(training_data, pd.DataFrame):
                    for feature in self.feature_names:
                        if feature in training_data.columns:
                            min_val = training_data[feature].min()
                            max_val = training_data[feature].max()
                            
                            # Add a small buffer (5%) to continuous features to allow exploration
                            feature_info = self.feature_types.get(feature, {'type': 'continuous'})
                            if feature_info['type'] == 'continuous':
                                range_size = max_val - min_val
                                buffer = range_size * 0.05  # 5% buffer
                                min_val = max(0, min_val - buffer) if min_val >= 0 else min_val - buffer
                                max_val = max_val + buffer
                            
                            feature_ranges[feature] = {
                                'min': min_val,
                                'max': max_val
                            }
                            print(f"[INFO] Feature '{feature}' range: {min_val:.4f} to {max_val:.4f}")
            
            for i, feature in enumerate(self.feature_names):
                # Feature name
                name_item = QTableWidgetItem(str(feature))
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.bounds_table.setItem(i, 0, name_item)
                
                # Get feature type info
                feature_info = self.feature_types.get(feature, {'type': 'continuous', 'categorical_values': None})
                feature_type = feature_info['type']
                categorical_values = feature_info['categorical_values']
                
                # Set bounds based on training data or use reasonable defaults
                if feature in feature_ranges:
                    # Use bounds from training data
                    min_val = str(feature_ranges[feature]['min'])
                    max_val = str(feature_ranges[feature]['max'])
                else:
                    # Use reasonable defaults based on feature type
                    if feature_type == "binary":
                        min_val, max_val = "0", "1"
                    elif feature_type == "categorical" and categorical_values:
                        min_val, max_val = str(min(categorical_values)), str(max(categorical_values))
                    else:
                        # Default range for continuous features when no data is available
                        min_val, max_val = "0.0", "100.0"
                
                # Min value
                min_item = QTableWidgetItem(min_val)
                self.bounds_table.setItem(i, 1, min_item)
                
                # Max value
                max_item = QTableWidgetItem(max_val)
                self.bounds_table.setItem(i, 2, max_item)
                
                # Fixed checkbox
                fixed_checkbox = QCheckBox()
                self.bounds_table.setCellWidget(i, 3, fixed_checkbox)
                
                # Feature type display
                type_item = QTableWidgetItem(feature_type)
                type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
                self.bounds_table.setItem(i, 4, type_item)
                
            self.bounds_table.resizeColumnsToContents()
            
            # Initialize enhanced bounds_dict
            self.feature_bounds = {}
            for feature in self.feature_names:
                feature_info = self.feature_types.get(feature, {'type': 'continuous', 'categorical_values': None})
                
                if feature in feature_ranges:
                    # Use bounds from training data
                    min_val = feature_ranges[feature]['min']
                    max_val = feature_ranges[feature]['max']
                elif feature_info['type'] == "binary":
                    min_val, max_val = 0.0, 1.0
                elif feature_info['type'] == "categorical" and feature_info['categorical_values']:
                    cat_values = feature_info['categorical_values']
                    min_val, max_val = float(min(cat_values)), float(max(cat_values))
                else:
                    # Default range for continuous features
                    min_val, max_val = 0.0, 100.0
                
                self.feature_bounds[feature] = FeatureBounds(
                    min_value=min_val,
                    max_value=max_val,
                    is_fixed=False,
                    feature_type=feature_info['type'],
                    categorical_values=feature_info['categorical_values']
                )
                
        except Exception as e:
            print(f"Error setting up feature bounds: {e}")
            self.status_label.setText(f"Error setting up feature bounds: {str(e)}")
    
    def get_feature_bounds(self):
        """Enhanced get feature bounds from table with type support"""
        bounds = {}
        
        print(f"[GET_BOUNDS] Reading bounds from table with {self.bounds_table.rowCount()} rows")
        
        # **CRITICAL FIX**: Process each row individually with error isolation
        for i in range(self.bounds_table.rowCount()):
            try:
                # Get feature name
                feature_item = self.bounds_table.item(i, 0)
                if feature_item is None:
                    continue
                feature = feature_item.text()
                
                # Get min value - handle categorical features safely
                min_item = self.bounds_table.item(i, 1)
                min_text = min_item.text() if min_item else ""
                try:
                    min_val = float(min_text) if min_text and min_text.strip() else 0.0
                except (ValueError, TypeError):
                    print(f"[GET_BOUNDS] Warning: Invalid min value '{min_text}' for {feature}, using 0.0")
                    min_val = 0.0
                
                # Get max value - handle categorical features safely  
                max_item = self.bounds_table.item(i, 2)
                max_text = max_item.text() if max_item else ""
                try:
                    max_val = float(max_text) if max_text and max_text.strip() else 1.0
                except (ValueError, TypeError):
                    print(f"[GET_BOUNDS] Warning: Invalid max value '{max_text}' for {feature}, using 1.0")
                    max_val = 1.0
                
                # Get fixed checkbox
                fixed_widget = self.bounds_table.cellWidget(i, 3)
                is_fixed = fixed_widget.isChecked() if fixed_widget else False
                
                # Get feature type info
                feature_info = self.feature_types.get(feature, {'type': 'continuous', 'categorical_values': None})
                feature_type = feature_info['type']
                categorical_values = feature_info['categorical_values']
                
                print(f"[GET_BOUNDS] Row {i}: {feature} = [{min_val}, {max_val}], fixed={is_fixed}, type={feature_type}")
                print(f"[GET_BOUNDS]   Raw table values: min_text='{min_text}', max_text='{max_text}'")
                
                # Handle fixed values
                if is_fixed:
                    fixed_val = min_val
                    print(f"[GET_BOUNDS] Setting fixed value for {feature}: {fixed_val}")
                    bounds[feature] = FeatureBounds(
                        min_value=fixed_val,
                        max_value=fixed_val,
                        is_fixed=True,
                        fixed_value=fixed_val,
                        feature_type=feature_type,
                        categorical_values=categorical_values
                    )
                else:
                    # Handle categorical features specially
                    if feature_type == 'categorical':
                        print(f"[GET_BOUNDS] Processing categorical feature {feature}")
                        # For categorical features, bounds should be [0, n-1] where n is number of categories
                        if categorical_values and len(categorical_values) > 0:
                            min_val = 0
                            max_val = len(categorical_values) - 1
                        else:
                            min_val = 0
                            max_val = 1  # Binary categorical default
                        print(f"[GET_BOUNDS] Categorical bounds for {feature}: [{min_val}, {max_val}]")
                    else:
                        # For continuous features, ensure valid bounds
                        if min_val >= max_val:
                            print(f"[GET_BOUNDS] Invalid bounds for {feature}: min={min_val}, max={max_val}, swapping")
                            if min_val == max_val:
                                max_val = min_val + 0.001
                            else:
                                min_val, max_val = max_val, min_val
                    
                    bounds[feature] = FeatureBounds(
                        min_value=min_val,
                        max_value=max_val,
                        is_fixed=False,
                        fixed_value=None,
                        feature_type=feature_type,
                        categorical_values=categorical_values
                    )
                    
                    print(f"[GET_BOUNDS] Final bounds for {feature}: [{min_val}, {max_val}]")
                
            except Exception as e:
                print(f"[GET_BOUNDS] Error processing row {i} (feature: {feature if 'feature' in locals() else 'unknown'}): {e}")
                # Continue with next row instead of failing completely
                if 'feature' in locals() and feature:
                    # Add default bounds for this feature
                    bounds[feature] = FeatureBounds(
                        min_value=0.0,
                        max_value=1.0,
                        is_fixed=False,
                        feature_type='continuous',
                        categorical_values=None
                    )
                continue
        
        # Ensure all features have bounds
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in bounds:
                    print(f"[GET_BOUNDS] Missing bounds for {feature}, adding default")
                    feature_info = self.feature_types.get(feature, {'type': 'continuous', 'categorical_values': None})
                    bounds[feature] = FeatureBounds(
                        min_value=0.0,
                        max_value=1.0,
                        is_fixed=False,
                        feature_type=feature_info['type'],
                        categorical_values=feature_info['categorical_values']
                    )
            
        return bounds
    
    def add_constraint(self):
        """Add a new constraint with enhanced validation and categorical feature handling"""
        if not self.feature_names:
            QMessageBox.warning(self, "Warning", "No features available for constraints!")
            return
            
        # Enhanced constraint dialog
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QListWidget, QCheckBox, QAbstractItemView
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Constraint")
        dialog.setModal(True)
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout(dialog)
        
        # Constraint type selection
        constraint_type_layout = QHBoxLayout()
        constraint_type_layout.addWidget(QLabel("Constraint Type:"))
        constraint_type_combo = QComboBox()
        constraint_type_combo.addItems(["Single Feature", "Multiple Features"])
        constraint_type_layout.addWidget(constraint_type_combo)
        layout.addLayout(constraint_type_layout)
        
        # Feature selection - stacked widgets
        feature_stack = QStackedWidget()
        
        # Single feature selection
        single_feature_widget = QWidget()
        single_feature_layout = QHBoxLayout(single_feature_widget)
        single_feature_layout.addWidget(QLabel("Feature:"))
        single_feature_combo = QComboBox()
        single_feature_combo.addItems(self.feature_names)
        single_feature_layout.addWidget(single_feature_combo)
        feature_stack.addWidget(single_feature_widget)
        
        # Feature type warning for single feature
        feature_type_warning = QLabel("")
        feature_type_warning.setStyleSheet("color: red; font-weight: bold;")
        single_feature_layout.addWidget(feature_type_warning)
        
        # Multiple feature selection
        multi_feature_widget = QWidget()
        multi_feature_layout = QVBoxLayout(multi_feature_widget)
        multi_feature_layout.addWidget(QLabel("Select Features and Coefficients:"))
        
        feature_coeff_table = QTableWidget()
        feature_coeff_table.setColumnCount(4)  # Added column for feature type
        feature_coeff_table.setHorizontalHeaderLabels(["Feature", "Include", "Coefficient", "Type"])
        feature_coeff_table.setRowCount(len(self.feature_names))
        
        for i, feature in enumerate(self.feature_names):
            # Feature name
            feature_item = QTableWidgetItem(feature)
            feature_item.setFlags(feature_item.flags() & ~Qt.ItemIsEditable)
            feature_coeff_table.setItem(i, 0, feature_item)
            
            # Include checkbox
            include_checkbox = QCheckBox()
            feature_coeff_table.setCellWidget(i, 1, include_checkbox)
            
            # Coefficient input
            coeff_spinbox = QDoubleSpinBox()
            coeff_spinbox.setRange(-100, 100)
            coeff_spinbox.setValue(1.0)
            coeff_spinbox.setSingleStep(0.1)
            feature_coeff_table.setCellWidget(i, 2, coeff_spinbox)
            
            # Feature type info
            feature_type = "continuous"
            if hasattr(self, 'feature_bounds') and feature in self.feature_bounds:
                bounds = self.feature_bounds[feature]
                feature_type = bounds.feature_type
                
            type_item = QTableWidgetItem(feature_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            # Highlight categorical features
            if feature_type in ["categorical", "binary"]:
                type_item.setForeground(QBrush(QColor("red")))
            feature_coeff_table.setItem(i, 3, type_item)
        
        feature_coeff_table.resizeColumnsToContents()
        multi_feature_layout.addWidget(feature_coeff_table)
        
        # Warning for categorical features in multi-feature constraints
        multi_warning_label = QLabel("Warning: Including categorical features in multi-feature constraints may cause unexpected behavior!")
        multi_warning_label.setStyleSheet("color: red; font-weight: bold;")
        multi_feature_layout.addWidget(multi_warning_label)
        
        feature_stack.addWidget(multi_feature_widget)
        layout.addWidget(feature_stack)
        
        # Connect constraint type change
        constraint_type_combo.currentIndexChanged.connect(feature_stack.setCurrentIndex)
        
        # Operator selection
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Operator:"))
        op_combo = QComboBox()
        op_combo.addItems(["<=", ">=", "=="])
        op_layout.addWidget(op_combo)
        layout.addLayout(op_layout)
        
        # Value input
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Bound Value:"))
        val_input = QDoubleSpinBox()
        val_input.setRange(-1000000, 1000000)
        val_input.setValue(0.5)
        val_input.setDecimals(6)  # Allow more precision
        val_layout.addWidget(val_input)
        layout.addLayout(val_layout)
        
        # Description
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Description:"))
        desc_input = QLineEdit()
        desc_input.setPlaceholderText("Optional description")
        desc_layout.addWidget(desc_input)
        layout.addLayout(desc_layout)
        
        # Show current feature bounds for reference
        bounds_info = QTextEdit()
        bounds_info.setReadOnly(True)
        bounds_info.setMaximumHeight(200)
        bounds_info.setPlaceholderText("Feature bounds information will appear here...")
        
        # Populate bounds info with more detailed information
        bounds_text = "Feature Bounds Reference:\n"
        for i, feature in enumerate(self.feature_names):
            if hasattr(self, 'feature_bounds') and feature in self.feature_bounds:
                bounds = self.feature_bounds[feature]
                bounds_text += f"• {feature} ({bounds.feature_type}):"
                
                if bounds.feature_type in ["categorical", "binary"]:
                    bounds_text += f" Values: {bounds.categorical_values}\n"
                else:
                    bounds_text += f" [{bounds.min_value:.4f}, {bounds.max_value:.4f}]\n"
            else:
                bounds_text += f"• {feature}: No bounds information available\n"
        
        bounds_info.setText(bounds_text)
        layout.addWidget(QLabel("Feature Bounds Reference:"))
        layout.addWidget(bounds_info)
        
        # Function to update feature type warning
        def update_feature_type_warning():
            feature_name = single_feature_combo.currentText()
            if hasattr(self, 'feature_bounds') and feature in self.feature_bounds:
                bounds = self.feature_bounds[feature_name]
                if bounds.feature_type in ["categorical", "binary"]:
                    feature_type_warning.setText(f"Warning: {feature_name} is {bounds.feature_type}!")
                    # Show possible values
                    if bounds.categorical_values:
                        val_input.setSpecialValueText(f"Possible values: {bounds.categorical_values}")
                else:
                    feature_type_warning.setText("")
                    val_input.setSpecialValueText("")
        
        # Connect signal
        single_feature_combo.currentIndexChanged.connect(update_feature_type_warning)
        
        # Initialize warning
        update_feature_type_warning()
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            # Process based on constraint type
            operator = op_combo.currentText()
            bound_value = val_input.value()
            
            if constraint_type_combo.currentIndex() == 0:
                # Single feature constraint
                feature_name = single_feature_combo.currentText()
                feature_idx = self.feature_names.index(feature_name)
                
                # Check if this is a categorical feature and validate bound value
                is_categorical = False
                if hasattr(self, 'feature_bounds') and feature_name in self.feature_bounds:
                    bounds = self.feature_bounds[feature_name]
                    if bounds.feature_type in ["categorical", "binary"]:
                        is_categorical = True
                        
                        # For categorical features, verify the bound makes sense
                        if bounds.categorical_values and bound_value not in bounds.categorical_values:
                            categorical_warning = (f"Warning: {bound_value} is not one of the possible values for "
                                                 f"categorical feature {feature_name}.\n\n"
                                                 f"Possible values: {bounds.categorical_values}\n\n"
                                                 f"Do you want to continue anyway?")
                            reply = QMessageBox.question(dialog, "Categorical Constraint Warning", 
                                                       categorical_warning, 
                                                       QMessageBox.Yes | QMessageBox.No)
                            if reply == QMessageBox.No:
                                return
                
                description = desc_input.text() or f"{feature_name} {operator} {bound_value}"
                
                # For categorical features, add special handling note to description
                if is_categorical:
                    description += f" (categorical feature)"
                
                constraint = OptimizationConstraint(
                    feature_indices=[feature_idx],
                    coefficients=[1.0],
                    operator=operator,
                    bound=bound_value,
                    description=description
                )
                
                self.constraints.append(constraint)
                
            else:
                # Multiple feature constraint
                feature_indices = []
                coefficients = []
                feature_names_list = []
                categorical_features = []
                
                for i in range(feature_coeff_table.rowCount()):
                    include_checkbox = feature_coeff_table.cellWidget(i, 1)
                    
                    if include_checkbox and include_checkbox.isChecked():
                        feature_name = feature_coeff_table.item(i, 0).text()
                        feature_idx = self.feature_names.index(feature_name)
                        feature_indices.append(feature_idx)
                        
                        coeff_spinbox = feature_coeff_table.cellWidget(i, 2)
                        coefficient = coeff_spinbox.value()
                        coefficients.append(coefficient)
                        
                        feature_names_list.append(f"{coefficient}*{feature_name}")
                        
                        # Check if this is a categorical feature
                        if hasattr(self, 'feature_bounds') and feature_name in self.feature_bounds:
                            bounds = self.feature_bounds[feature_name]
                            if bounds.feature_type in ["categorical", "binary"]:
                                categorical_features.append(feature_name)
                
                if not feature_indices:
                    QMessageBox.warning(self, "Warning", "No features selected for constraint!")
                    return
                
                # Warn about categorical features in multi-feature constraints
                if categorical_features:
                    categorical_warning = (f"Warning: The following categorical features are included "
                                         f"in this multi-feature constraint: {', '.join(categorical_features)}.\n\n"
                                         f"Constraints on categorical features may not work as expected.\n\n"
                                         f"Do you want to continue anyway?")
                    reply = QMessageBox.question(dialog, "Categorical Constraint Warning", 
                                               categorical_warning, 
                                               QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.No:
                        return
                
                # Create description if not provided
                if not desc_input.text():
                    description = " + ".join(feature_names_list) + f" {operator} {bound_value}"
                    if categorical_features:
                        description += f" (includes categorical features: {', '.join(categorical_features)})"
                else:
                    description = desc_input.text()
                
                constraint = OptimizationConstraint(
                    feature_indices=feature_indices,
                    coefficients=coefficients,
                    operator=operator,
                    bound=bound_value,
                    description=description
                )
                
                self.constraints.append(constraint)
            
            # Update the constraints display
            self.update_constraints_table()
            
            # Print detailed debug info about the new constraint
            print(f"\n[CONSTRAINT] Added new constraint: {constraint.description}")
            print(f"[CONSTRAINT] Feature indices: {constraint.feature_indices}")
            print(f"[CONSTRAINT] Coefficients: {constraint.coefficients}")
            print(f"[CONSTRAINT] Operator: {constraint.operator}")
            print(f"[CONSTRAINT] Bound: {constraint.bound}")
            
            # Show success message with debug info
            QMessageBox.information(self, "Constraint Added", 
                                  f"Successfully added constraint:\n{constraint.description}\n\n"
                                  f"Feature indices: {constraint.feature_indices}\n"
                                  f"Make sure to check the console for detailed constraint evaluation during optimization.")
    
    def update_constraints_table(self):
        """Update constraints table display"""
        self.constraints_table.setRowCount(len(self.constraints))
        
        for i, constraint in enumerate(self.constraints):
            # Features
            feature_names = [self.feature_names[idx] for idx in constraint.feature_indices if idx < len(self.feature_names)]
            features_text = ", ".join(feature_names)
            self.constraints_table.setItem(i, 0, QTableWidgetItem(features_text))
            
            # Operator
            self.constraints_table.setItem(i, 1, QTableWidgetItem(constraint.operator))
            
            # Value
            self.constraints_table.setItem(i, 2, QTableWidgetItem(str(constraint.bound)))
            
            # Description
            self.constraints_table.setItem(i, 3, QTableWidgetItem(constraint.description))
        
        self.constraints_table.resizeColumnsToContents()
        
    def start_optimization(self):
        """Start optimization process"""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model available for optimization!")
            return
            
        try:
            # Get configuration
            algorithm = self.algorithm_combo.currentText()
            target_direction = self.target_combo.currentText()  # Already lowercase
            bounds = self.get_feature_bounds()
            
            # Enhanced algorithm-specific configuration
            config = {
                # General parameters
                'max_iterations': self.n_iterations_spin.value(),
                'n_iterations': self.n_iterations_spin.value(),
                'n_points_per_dim': self.n_points_spin.value(),
                
                # Genetic Algorithm parameters
                'max_generations': self.n_iterations_spin.value(),
                'population_size': self.population_spin.value(),
                'mutation_rate': self.mutation_rate_spin.value(),
                'crossover_rate': self.crossover_rate_spin.value(),
                'elite_size': max(1, self.population_spin.value() // 10),
                
                # Particle Swarm Optimization parameters
                'n_particles': self.population_spin.value(),
                'inertia_weight': self.inertia_weight_spin.value(),
                'cognitive_param': 2.0,
                'social_param': 2.0,
                
                # Scipy optimization parameters
                'maxiter': self.n_iterations_spin.value(),
                'popsize': 15,
                'niter': self.n_iterations_spin.value(),
                'n_calls': self.n_iterations_spin.value(),
                
                # COBYLA parameters
                'rhobeg': self.trust_radius_spin.value(),
                'rhoend': self.trust_radius_spin.value() * 1e-6,
                'objective_scale_estimate': 500.0,  # Estimated scale for constraint penalties
                
                # Performance settings
                'parallel_processing': True,
                'real_time_updates': self.realtime_viz_checkbox.isChecked(),
                'performance_tracking': self.performance_viz_checkbox.isChecked()
            }
            
            # Log feature names and constraints for debugging
            print("\n[OPTIMIZATION START] Feature order reference:")
            for i, feature_name in enumerate(self.feature_names):
                print(f"  {i}: {feature_name}")
                
            # **CRITICAL FIX**: Log categorical features and their original values
            if hasattr(self, 'original_categorical_values') and self.original_categorical_values:
                print("\n[OPTIMIZATION START] Categorical features:")
                for feature_name, original_values in self.original_categorical_values.items():
                    if feature_name in self.feature_names:
                        idx = self.feature_names.index(feature_name)
                        print(f"  {idx}: {feature_name} - Original values: {original_values}")
                        
                        # Update feature type in bounds if needed
                        if feature_name in bounds:
                            bounds[feature_name].feature_type = "categorical"
                            # Set categorical_values to numeric range (0 to n-1)
                            bounds[feature_name].categorical_values = list(range(len(original_values)))
                            print(f"    Optimizer will use values: {bounds[feature_name].categorical_values}")
                
            print("\n[OPTIMIZATION START] Constraints:")
            for i, constraint in enumerate(self.constraints):
                feature_indices = constraint.feature_indices
                feature_names = [self.feature_names[idx] if idx < len(self.feature_names) else f"unknown_{idx}" for idx in feature_indices]
                print(f"  Constraint {i+1}: {constraint.description}")
                print(f"    Features: {', '.join([f'{name}(idx={idx})' for name, idx in zip(feature_names, feature_indices)])}")
                print(f"    Operator: {constraint.operator}, Bound: {constraint.bound}")
                
            # Reset real-time history
            self.real_time_history = []
            
            # **DEBUG**: Print bounds before passing to worker
            print(f"\n[START_OPT] Bounds being passed to worker:")
            for feature_name, bound_obj in bounds.items():
                print(f"[START_OPT]   {feature_name}: [{bound_obj.min_value}, {bound_obj.max_value}] (type: {bound_obj.feature_type})")
            
            # Create and start worker thread
            self.worker = OptimizationWorker(
                self.model, bounds, self.constraints, 
                algorithm, config, target_direction, self.feature_names
            )
            
            # **CRITICAL FIX**: Pass original categorical values to worker
            if hasattr(self, 'original_categorical_values') and self.original_categorical_values:
                # Initialize categorical mappings in the worker
                for feature_name, original_values in self.original_categorical_values.items():
                    if feature_name in self.feature_names and feature_name in bounds:
                        # Create mapping between original values and optimizer's numeric values
                        if feature_name not in self.worker.categorical_mappings:
                            self.worker.categorical_mappings[feature_name] = {
                                'orig_to_num': {},
                                'num_to_orig': {}
                            }
                            
                        # Fill the mappings
                        for i, orig_value in enumerate(original_values):
                            self.worker.categorical_mappings[feature_name]['orig_to_num'][orig_value] = i
                            self.worker.categorical_mappings[feature_name]['num_to_orig'][i] = orig_value
                            
                        print(f"[CATEGORICAL] Created mapping for {feature_name}: {self.worker.categorical_mappings[feature_name]['num_to_orig']}")
            
            # Connect signals
            self.worker.progress_updated.connect(self.progress_bar.setValue)
            self.worker.status_updated.connect(self.status_label.setText)
            self.worker.optimization_completed.connect(self.on_optimization_completed)
            self.worker.error_occurred.connect(self.on_optimization_error)
            self.worker.iteration_completed.connect(self.on_iteration_completed)
            self.worker.real_time_data.connect(self.on_real_time_data)
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Clear previous plot if real-time is enabled
            if self.realtime_viz_checkbox.isChecked():
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Starting optimization...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas.draw()
            
            # Start optimization
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start optimization: {str(e)}")
            
    def stop_optimization(self):
        """Stop optimization process"""
        if self.worker and self.worker.isRunning():
            self.worker.stop_optimization()
            self.worker.wait()
            
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Optimization stopped.")
        
    def on_iteration_completed(self, iteration, best_value, best_params, extra_data):
        """Handle iteration completion with throttled real-time visualization updates"""
        try:
            # Store current best for real-time updates
            if not hasattr(self, 'real_time_history'):
                self.real_time_history = []
            
            self.real_time_history.append({
                'iteration': iteration,
                'best_value': best_value,
                'best_params': best_params,
                'timestamp': time.time(),
                'extra_data': extra_data
            })
            
            # **STACK OVERFLOW FIX**: Further reduce visualization update frequency
            # Update visualization only if enabled and very infrequently (every 25 iterations)
            if (hasattr(self, 'realtime_viz_checkbox') and 
                self.realtime_viz_checkbox.isChecked() and 
                len(self.real_time_history) % 25 == 0):  # Changed from 10 to 25
                try:
                    self._update_real_time_plot()
                except Exception:
                    pass  # Silently ignore visualization errors
        except Exception:
            pass  # Silently ignore any errors in iteration handling
    
    def on_real_time_data(self, data):
        """Handle real-time optimization data with throttled advanced visualization"""
        try:
            # **STACK OVERFLOW FIX**: Add throttling and reduce processing frequency
            if (hasattr(self, 'realtime_viz_checkbox') and 
                self.realtime_viz_checkbox.isChecked()):
                
                # Only update visualization every 15 iterations to reduce load
                iteration = data.get('iteration', 0)
                if iteration % 15 == 0:  # Throttle advanced visualization updates
                    try:
                        self._update_enhanced_real_time_plot(data)
                    except Exception:
                        pass  # Silently ignore visualization errors
                
                # Update status more frequently but with error protection
                if iteration % 5 == 0:  # Update status every 5 iterations
                    try:
                        if 'extra_data' in data and data['extra_data']:
                            extra = data['extra_data']
                            if 'diversity' in extra:
                                diversity_text = f" | Diversity: {extra['diversity']:.4f}"
                            else:
                                diversity_text = ""
                            
                            self.status_label.setText(
                                f"Iteration {data['iteration']}: Best = {data['best_value']:.6f}{diversity_text}"
                            )
                    except Exception:
                        pass  # Silently ignore status update errors
        except Exception:
            pass  # Silently ignore all real-time update errors
        
    def on_optimization_completed(self, results):
        """Handle optimization completion with enhanced stability and memory cleanup"""
        try:
            self.optimization_results = results
            
            # **CRITICAL**: Stop worker and clean up resources first
            if hasattr(self, 'worker') and self.worker:
                if self.worker.isRunning():
                    self.worker.stop_optimization()
                    self.worker.wait(3000)  # Wait max 3 seconds
                    if self.worker.isRunning():
                        self.worker.terminate()  # Force terminate if needed
                        self.worker.wait(1000)
                
            # **MEMORY CLEANUP**: Force garbage collection
            gc.collect()
            
            # Update UI with error protection
            try:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.progress_bar.setVisible(False)
            except Exception:
                pass  # UI state errors are non-critical
            
            # **BOUNDS VERIFICATION**: Check if results violate bounds
            try:
                self._verify_results_bounds(results)
            except Exception as e:
                print(f"Warning: Bounds verification failed: {e}")

            # **SAFE RESULT DISPLAY**: Wrap in try-catch to prevent crashes
            try:
                self.display_results(results)
            except Exception as e:
                print(f"Warning: Results display failed: {e}")
                # Create minimal fallback display
                try:
                    self.results_text.setText(f"Optimization completed.\nBest value: {results.get('best_value', 'N/A')}")
                except Exception:
                    pass
            
            # **SAFE VISUALIZATION**: Wrap in try-catch with fallback
            try:
                # Clear any existing plots first
                self.figure.clear()
                plt.close('all')
                gc.collect()  # Clean up matplotlib objects
                
                # Create visualizations with timeout protection
                self.create_visualizations(results)
            except Exception as e:
                print(f"Warning: Visualization creation failed: {e}")
                # Create minimal fallback plot
                try:
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, f'Optimization Completed\nBest Value: {results.get("best_value", "N/A")}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title('Optimization Results')
                    self.canvas.draw()
                except Exception:
                    pass
            
            # **FINAL CLEANUP**: Clear worker reference and trigger GC
            try:
                if hasattr(self, 'worker'):
                    self.worker = None
                gc.collect()
            except Exception:
                pass
            
            # Emit signal with error protection
            try:
                self.optimization_completed.emit(results)
            except Exception:
                pass  # Signal emission errors are non-critical
                
        except Exception as e:
            # **ULTIMATE FALLBACK**: If everything fails, at least restore UI state
            print(f"Critical error in optimization completion: {e}")
            try:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.progress_bar.setVisible(False)
                self.status_label.setText("Optimization completed with errors.")
            except Exception:
                pass

    def display_results(self, results):
        """Display enhanced optimization results with memory protection"""
        try:
            # **MEMORY PROTECTION**: Limit result data size
            if not results or not isinstance(results, dict):
                self.results_text.setText("Error: Invalid results data")
                return
            
            # Calculate additional metrics with error protection
            execution_time = results.get('execution_time', 0)
            eval_per_sec = results.get('evaluations_per_second', 0)
            
            # **SIMPLIFIED DISPLAY**: Reduce memory usage by limiting text length
            text = f"""Target Optimization Results - {results.get('algorithm', 'Unknown')}
{'=' * 50}

Best Solution:
Target Direction: {results.get('target_direction', 'Unknown').title()}
Best Value: {results.get('best_value', 'N/A'):.6f}

Optimal Parameters:
"""
            
            # **ENHANCED PARAMETER DISPLAY**: Show both normalized and original values
            best_params = results.get('best_params', {})
            model_wrapper = results.get('model_wrapper')
            
            if isinstance(best_params, dict):
                # Try to get original scale parameters
                original_params = None
                if model_wrapper and hasattr(model_wrapper, 'inverse_transform_features'):
                    try:
                        # Create a single row DataFrame/array for inverse transform
                        import pandas as pd
                        import numpy as np
                        
                        # Convert best_params to the format expected by inverse_transform
                        param_array = np.array([list(best_params.values())]).reshape(1, -1)
                        
                        # Try inverse transform
                        original_data = model_wrapper.inverse_transform_features(param_array)
                        
                        if isinstance(original_data, pd.DataFrame):
                            original_params = original_data.iloc[0].to_dict()
                        elif isinstance(original_data, np.ndarray):
                            feature_names = list(best_params.keys())
                            original_params = dict(zip(feature_names, original_data[0]))
                            
                    except Exception as e:
                        original_params = None
                
                param_count = 0
                for feature, normalized_value in best_params.items():
                    if param_count >= 20:  # Limit to 20 parameters max
                        text += "  • ... (more parameters omitted)\n"
                        break
                    
                    try:
                        if original_params and feature in original_params:
                            original_value = original_params[feature]
                            # Values should be the same now since bounds are in original scale
                            text += f"  • {feature}: {original_value:.6f}\n"
                        else:
                            # Show the value directly (now in original scale)
                            text += f"  • {feature}: {normalized_value:.6f}\n"
                        param_count += 1
                    except Exception:
                        text += f"  • {feature}: {normalized_value}\n"
                        param_count += 1
            
            text += f"""
Optimization Statistics:
  • Total Iterations: {results.get('n_iterations', 'N/A')}
  • Constraint Violations: {results.get('constraint_violations', 'N/A')}
  • Constraints Satisfied: {'Yes' if results.get('constraints_satisfied', False) else 'No'}

Performance Metrics:
  • Execution Time: {execution_time:.2f} seconds
  • Evaluation Rate: {eval_per_sec:.1f} eval/sec
  • Status: Completed
"""
            
            # **SAFE TEXT SETTING**: Limit total text length
            if len(text) > 10000:  # Limit to 10KB of text
                text = text[:10000] + "\n... (output truncated)"
            
            self.results_text.setText(text)
            
        except Exception as e:
            # **FALLBACK DISPLAY**: Minimal safe display
            try:
                fallback_text = f"Optimization Completed\nBest Value: {results.get('best_value', 'N/A')}\nStatus: Success"
                self.results_text.setText(fallback_text)
            except Exception:
                self.results_text.setText("Optimization Completed - Display Error")

    def create_visualizations(self, results):
        """Create enhanced interactive optimization visualizations with stability protection"""
        try:
            # **MEMORY PROTECTION**: Clear and prepare figure
            self.figure.clear()
            plt.close('all')
            gc.collect()
            
            # **ENHANCED DATA SOURCE**: Use real_time_history as primary source
            real_time_history = results.get('real_time_history', [])
            legacy_history = results.get('history', [])
            
            # Determine which history to use
            if real_time_history:
                # Use real-time history (preferred)
                history_data = real_time_history
                iterations = [point.get('iteration', i) for i, point in enumerate(history_data)]
                objective_values = [point.get('best_value', 0) for point in history_data]
                param_history = [point.get('best_params', []) for point in history_data]
            elif legacy_history:
                # Use legacy history format
                iterations = list(range(len(legacy_history)))
                param_history = [point[0] if len(point) > 0 else [] for point in legacy_history]
                objective_values = [point[1] if len(point) > 1 else 0 for point in legacy_history]
            else:
                # No history data - create success indicator only
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Optimization Complete\nBest Value: {results.get("best_value", "N/A"):.6f}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=16)
                ax.set_title('Optimization Results')
                self.figure.tight_layout()
                self.canvas.draw()
                return
                
            # **DATA VALIDATION**: Ensure we have valid data
            if not objective_values:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Optimization Complete\nBest Value: {results.get("best_value", "N/A"):.6f}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=16)
                ax.set_title('Optimization Results')
                self.figure.tight_layout()
                self.canvas.draw()
                return
                
            # **DATA PREPARATION**: Limit data size to prevent memory issues
            max_points = 500
            if len(objective_values) > max_points:
                step = len(objective_values) // max_points
                iterations = iterations[::step]
                objective_values = objective_values[::step]
                param_history = param_history[::step]
            
            feature_names = results.get('feature_names', [])
            
            # **ENHANCED PLOTTING**: Create comprehensive visualization
            if len(feature_names) <= 2 and param_history and param_history[0]:
                # 2D parameter space + convergence plot
                fig_rows = 2
                fig_cols = 1
                
                # Convergence plot
                ax1 = self.figure.add_subplot(fig_rows, fig_cols, 1)
                ax1.plot(iterations, objective_values, 'b-', linewidth=2, alpha=0.8, label='Convergence')
                
                # Highlight best point
                if results.get('best_value') is not None:
                    best_iter = iterations[-1] if iterations else len(objective_values) - 1
                    ax1.scatter([best_iter], [objective_values[-1]], color='red', s=80, 
                              label=f'Best: {results["best_value"]:.3f}', zorder=5)
                
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Objective Value')
                ax1.set_title('Optimization Convergence')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Parameter evolution plot
                if len(feature_names) == 2 and param_history:
                    ax2 = self.figure.add_subplot(fig_rows, fig_cols, 2)
                    
                    # Extract parameter values
                    param1_vals = []
                    param2_vals = []
                    for params in param_history:
                        if isinstance(params, dict):
                            param1_vals.append(params.get(feature_names[0], 0))
                            param2_vals.append(params.get(feature_names[1], 0))
                        elif isinstance(params, (list, tuple)) and len(params) >= 2:
                            param1_vals.append(params[0])
                            param2_vals.append(params[1])
                    
                    if param1_vals and param2_vals:
                        # Plot parameter evolution path
                        ax2.plot(param1_vals, param2_vals, 'g-', alpha=0.6, linewidth=1, label='Path')
                        ax2.scatter(param1_vals, param2_vals, c=iterations, cmap='viridis', s=30, alpha=0.7)
                        
                        # Highlight start and end
                        ax2.scatter([param1_vals[0]], [param2_vals[0]], color='green', s=100, 
                                   marker='o', label='Start', zorder=5)
                        ax2.scatter([param1_vals[-1]], [param2_vals[-1]], color='red', s=100, 
                                   marker='*', label='Best', zorder=5)
                        
                        ax2.set_xlabel(feature_names[0])
                        ax2.set_ylabel(feature_names[1])
                        ax2.set_title('Parameter Space Exploration')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                
            else:
                # Single convergence plot for multi-dimensional problems
                ax = self.figure.add_subplot(111)
                
                # Enhanced convergence plot
                ax.plot(iterations, objective_values, 'b-', linewidth=2, alpha=0.8, label='Convergence')
                
                # Add moving average for smoother visualization
                if len(objective_values) > 10:
                    window_size = min(10, len(objective_values) // 5)
                    if window_size > 1:
                        moving_avg = []
                        for i in range(len(objective_values)):
                            start_idx = max(0, i - window_size // 2)
                            end_idx = min(len(objective_values), i + window_size // 2 + 1)
                            avg = np.mean(objective_values[start_idx:end_idx])
                            moving_avg.append(avg)
                        
                        ax.plot(iterations, moving_avg, 'r--', linewidth=1, alpha=0.7, 
                               label=f'Trend (window={window_size})')
                
                # Highlight best point
                if results.get('best_value') is not None:
                    best_iter = iterations[-1] if iterations else len(objective_values) - 1
                    ax.scatter([best_iter], [objective_values[-1]], color='red', s=100, 
                              marker='*', label=f'Best: {results["best_value"]:.3f}', zorder=5)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Objective Value')
                ax.set_title(f'Optimization Convergence ({len(feature_names)} features)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics annotation
                if objective_values:
                    improvement = abs(objective_values[-1] - objective_values[0]) if len(objective_values) > 1 else 0
                    ax.text(0.02, 0.98, f'Improvement: {improvement:.3f}\nIterations: {len(objective_values)}', 
                           transform=ax.transAxes, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # **SAFE DRAWING**: Use error protection
            try:
                self.figure.tight_layout()
                self.canvas.draw_idle()  # Use draw_idle for better performance
            except Exception:
                # If drawing fails, just show text
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Visualization Failed\nPlease check text results', 
                       ha='center', va='center', transform=ax.transAxes)
                self.canvas.draw()
        
        except Exception as e:
            # **ULTIMATE FALLBACK**: Show error message plot
            try:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Visualization Error\nOptimization Complete\nBest Value: {results.get("best_value", "N/A")}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Optimization Results')
                self.canvas.draw()
            except Exception:
                pass  # If even this fails, give up on visualization

    def _verify_results_bounds(self, results):
        """Verify that optimization results satisfy feature bounds"""
        try:
            best_params = results.get('best_params', {})
            if not isinstance(best_params, dict) or not best_params:
                print("[BOUNDS_CHECK] No parameters to verify")
                return
                
            # **CRITICAL**: Get FRESH bounds from UI to ensure we use the real bounds, not cached ones
            try:
                # Read bounds directly from the interface table
                ui_bounds = {}
                for i in range(self.bounds_table.rowCount()):
                    feature_item = self.bounds_table.item(i, 0)
                    if feature_item is None:
                        continue
                    feature = feature_item.text()
                    
                    min_item = self.bounds_table.item(i, 1)
                    max_item = self.bounds_table.item(i, 2)
                    
                    try:
                        min_val = float(min_item.text()) if min_item and min_item.text() else 0.0
                        max_val = float(max_item.text()) if max_item and max_item.text() else 1.0
                        ui_bounds[feature] = (min_val, max_val)
                        print(f"[BOUNDS_CHECK] UI bounds for {feature}: [{min_val}, {max_val}]")
                    except (ValueError, TypeError):
                        # For categorical features, use [0, 1] bounds
                        ui_bounds[feature] = (0.0, 1.0)
                        print(f"[BOUNDS_CHECK] UI bounds for {feature}: [0.0, 1.0] (categorical/default)")
                        
                current_bounds = ui_bounds
            except Exception as e:
                print(f"[BOUNDS_CHECK] Error reading UI bounds: {e}")
                # Fallback to get_feature_bounds
                current_bounds_obj = self.get_feature_bounds()
                current_bounds = {name: (bounds.min_value, bounds.max_value) 
                                for name, bounds in current_bounds_obj.items()}
            
            print(f"\n[BOUNDS_CHECK] ===== FINAL BOUNDS VERIFICATION =====")
            print(f"[BOUNDS_CHECK] Checking {len(best_params)} optimized parameters")
            
            violations = []
            for feature_name, optimized_value in best_params.items():
                if feature_name in current_bounds:
                    min_val, max_val = current_bounds[feature_name]
                    
                    # Check bounds violation
                    if optimized_value < min_val or optimized_value > max_val:
                        violation_msg = f"{feature_name}: {optimized_value:.6f} NOT in [{min_val:.6f}, {max_val:.6f}]"
                        violations.append(violation_msg)
                        print(f"[BOUNDS_CHECK] ❌ {violation_msg}")
                    else:
                        print(f"[BOUNDS_CHECK] ✅ {feature_name}: {optimized_value:.6f} in [{min_val:.6f}, {max_val:.6f}]")
                else:
                    print(f"[BOUNDS_CHECK] ⚠️  {feature_name}: No bounds definition found")
            
            if violations:
                print(f"[BOUNDS_CHECK] 🚨 CRITICAL: Found {len(violations)} bounds violations!")
                print("[BOUNDS_CHECK] This indicates a serious problem in the optimization algorithm!")
                for i, violation in enumerate(violations[:5]):  # Show first 5 violations
                    print(f"[BOUNDS_CHECK]   {i+1}. {violation}")
                if len(violations) > 5:
                    print(f"[BOUNDS_CHECK]   ... and {len(violations)-5} more violations")
            else:
                print(f"[BOUNDS_CHECK] ✅ SUCCESS: All {len(best_params)} parameters satisfy bounds")
                
            print(f"[BOUNDS_CHECK] ==========================================\n")
            
        except Exception as e:
            print(f"[BOUNDS_CHECK] ERROR during verification: {e}")

    def reset(self):
        """Reset the module with enhanced cleanup"""
        try:
            # **SAFE WORKER CLEANUP**: Force stop and cleanup
            if hasattr(self, 'worker') and self.worker:
                if self.worker.isRunning():
                    self.worker.stop_optimization()
                    self.worker.wait(2000)  # Wait max 2 seconds
                    if self.worker.isRunning():
                        self.worker.terminate()  # Force terminate
                        self.worker.wait(1000)
                self.worker = None
            
            # **MEMORY CLEANUP**: Clear all data structures
            self.model = None
            self.feature_names = None
            self.feature_bounds = {}
            self.optimization_results = None
            
            # Clear real-time history safely
            if hasattr(self, 'real_time_history'):
                try:
                    self.real_time_history.clear()
                except Exception:
                    self.real_time_history = []
            
            # **SAFE UI RESET**: Protect against UI errors
            try:
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(False)
                self.progress_bar.setVisible(False)
                self.progress_bar.setValue(0)
            except Exception:
                pass
            
            try:
                self.bounds_table.setRowCount(0)
            except Exception:
                pass
                
            try:
                self.results_text.clear()
            except Exception:
                pass
            
            # **SAFE PLOT CLEANUP**: Clear with multiple fallbacks
            try:
                self.figure.clear()
                plt.close('all')
                self.canvas.draw()
            except Exception:
                try:
                    self.figure.clear()
                    self.canvas.draw()
                except Exception:
                    pass
            
            # **FORCE GARBAGE COLLECTION**: Clean up memory
            gc.collect()
            
            try:
                self.status_label.setText("Ready")
            except Exception:
                pass
                
        except Exception as e:
            print(f"Warning: Reset encountered errors: {e}")
            # Force minimal cleanup even if errors occur
            try:
                gc.collect()
            except Exception:
                pass
        
    def on_optimization_error(self, error_msg):
        """Handle optimization error"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Optimization Error", error_msg)
        self.status_label.setText(f"Error: {error_msg}")
        
    def export_results(self):
        """Export optimization results"""
        if not self.optimization_results:
            QMessageBox.warning(self, "Warning", "No results to export!")
            return
            
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Optimization Results", 
                f"optimization_results_{self.optimization_results['algorithm'].lower().replace(' ', '_')}.txt",
                "Text Files (*.txt);;All Files (*)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.results_text.toPlainText())
                    
                QMessageBox.information(self, "Success", f"Results exported to {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
            
    def _update_real_time_plot(self):
        """Update basic real-time plot during optimization"""
        if not hasattr(self, 'real_time_history') or not self.real_time_history:
            return
            
        try:
            # Clear and create a simple convergence plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            iterations = [h['iteration'] for h in self.real_time_history]
            best_values = [h['best_value'] for h in self.real_time_history]
            
            ax.plot(iterations, best_values, 'b-', linewidth=2, alpha=0.8)
            ax.scatter(iterations[-1], best_values[-1], color='red', s=100, zorder=5)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Objective Value')
            ax.set_title('Real-time Optimization Progress')
            ax.grid(True, alpha=0.3)
            
            # Add current best value as text
            ax.text(0.02, 0.98, f'Current Best: {best_values[-1]:.6f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            pass  # Silently ignore plotting errors during optimization
    
    def _update_enhanced_real_time_plot(self, data):
        """Update enhanced real-time plot with algorithm-specific information and stack overflow protection"""
        try:
            if not data.get('history'):
                return
            
            # **STACK OVERFLOW FIX**: Add matplotlib state cleanup and error handling
            # Clear figure with proper cleanup
            try:
                self.figure.clear()
                plt.close('all')  # Clean up any dangling matplotlib objects
            except Exception:
                return  # Skip update if cleanup fails
            
            # Determine layout based on available data (simplified)
            extra_data = data.get('extra_data', {})
            has_population = 'population_fitness' in extra_data
            
            # **Simplified layout to reduce complexity and memory usage**
            if has_population:
                ax1 = self.figure.add_subplot(121)
                ax2 = self.figure.add_subplot(122)
                ax3 = None
            else:
                # Single plot for basic convergence
                ax1 = self.figure.add_subplot(111)
                ax2 = ax3 = None
            
            # Main convergence plot with simplified rendering
            history = data['history']
            if len(history) > 100:  # Limit history to prevent memory issues
                history = history[-100:]
                
            iterations = [h['iteration'] for h in history]
            best_values = [h['best_value'] for h in history]
            
            if iterations and best_values:
                ax1.plot(iterations, best_values, 'b-', linewidth=1, alpha=0.8)  # Reduced linewidth
                if iterations:
                    ax1.scatter(iterations[-1], best_values[-1], color='red', s=50, zorder=5)  # Reduced size
            
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Best Objective Value')
            ax1.set_title('Convergence Progress')
            ax1.grid(True, alpha=0.3)
            
            # Add performance text (simplified)
            if best_values:
                current_best = best_values[-1]
                ax1.text(0.02, 0.98, f'Current Best: {current_best:.6f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Simplified population fitness distribution (for GA/PSO)
            if ax2 and has_population:
                try:
                    fitness_values = extra_data.get('population_fitness', [])
                    if hasattr(fitness_values, '__iter__') and len(fitness_values) > 1:
                        # Limit bins and simplify histogram
                        bins = min(10, len(fitness_values)//3)  # Reduced bins
                        ax2.hist(fitness_values, bins=bins, alpha=0.7, color='lightgreen')
                        ax2.set_xlabel('Fitness Value')
                        ax2.set_ylabel('Frequency')
                        ax2.set_title('Population Fitness')
                        ax2.grid(True, alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, 'Single Fitness Value', ha='center', va='center', 
                                transform=ax2.transAxes)
                except Exception:
                    ax2.text(0.5, 0.5, 'Fitness Data Error', ha='center', va='center', 
                            transform=ax2.transAxes)
            
            # **Safe drawing with error handling**
            try:
                self.figure.tight_layout()
                self.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
            except Exception:
                pass  # Skip drawing if it fails
                
        except Exception:
            pass  # Silently ignore all plotting errors during optimization
        