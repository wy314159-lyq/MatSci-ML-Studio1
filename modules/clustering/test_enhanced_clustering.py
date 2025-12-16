"""
Comprehensive Unit Tests for Enhanced Clustering Module
Tests all new features and algorithms
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all new clustering modules
from modules.clustering.advanced_algorithms import (
    FuzzyCMeans, ConsensusKMeans, MiniBatchKMeansPlus, EnhancedAffinityPropagation
)
from modules.clustering.adaptive_optimization import (
    BayesianClusterOptimizer, AdaptiveParameterRecommender, AdvancedParameterOptimizer
)
from modules.clustering.evaluation_metrics import ComprehensiveClusteringEvaluator
from modules.clustering.algorithm_selection import (
    DataCharacterizationEngine, AlgorithmSelector, IntelligentAlgorithmSelector
)
from modules.clustering.interpretability import (
    ClusterProfiler, ClusterExplainer, ClusterVisualExplainer, ComprehensiveClusteringInterpreter
)
from modules.clustering.performance_optimization import (
    MemoryEfficientScaler, StreamingKMeans, ApproximateDBSCAN, ScalableClusteringPipeline
)

warnings.filterwarnings('ignore')


class TestAdvancedAlgorithms(unittest.TestCase):
    """Test advanced clustering algorithms"""
    
    def setUp(self):
        """Set up test data"""
        self.X_blobs, self.y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                                              random_state=42, cluster_std=1.0)
        self.X_moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
        
    def test_fuzzy_cmeans(self):
        """Test Fuzzy C-Means implementation"""
        fcm = FuzzyCMeans(n_clusters=4, random_state=42)
        labels = fcm.fit_predict(self.X_blobs)
        
        # Basic checks
        self.assertEqual(len(labels), len(self.X_blobs))
        self.assertEqual(len(np.unique(labels)), 4)
        self.assertTrue(hasattr(fcm, 'membership_matrix_'))
        self.assertTrue(hasattr(fcm, 'cluster_centers_'))
        
        # Check membership matrix properties
        membership = fcm.membership_matrix_
        self.assertEqual(membership.shape, (len(self.X_blobs), 4))
        
        # Each row should sum to 1 (approximately)
        row_sums = np.sum(membership, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
        
        # All values should be between 0 and 1
        self.assertTrue(np.all(membership >= 0))
        self.assertTrue(np.all(membership <= 1))
    
    def test_consensus_kmeans(self):
        """Test Consensus K-Means implementation"""
        # Test voting consensus
        ck_voting = ConsensusKMeans(n_clusters=4, n_estimators=5, 
                                   consensus_method='voting', random_state=42)
        labels_voting = ck_voting.fit_predict(self.X_blobs)
        
        self.assertEqual(len(labels_voting), len(self.X_blobs))
        self.assertEqual(len(np.unique(labels_voting)), 4)
        
        # Test co-occurrence consensus
        ck_cooc = ConsensusKMeans(n_clusters=4, n_estimators=5, 
                                 consensus_method='co-occurrence', random_state=42)
        labels_cooc = ck_cooc.fit_predict(self.X_blobs)
        
        self.assertEqual(len(labels_cooc), len(self.X_blobs))
        
        # Test evidence accumulation
        ck_ea = ConsensusKMeans(n_clusters=4, n_estimators=5, 
                               consensus_method='evidence_accumulation', random_state=42)
        labels_ea = ck_ea.fit_predict(self.X_blobs)
        
        self.assertEqual(len(labels_ea), len(self.X_blobs))
    
    def test_minibatch_kmeans_plus(self):
        """Test Enhanced Mini-Batch K-Means"""
        mbk = MiniBatchKMeansPlus(n_clusters=4, batch_size='auto', random_state=42)
        labels = mbk.fit_predict(self.X_blobs)
        
        self.assertEqual(len(labels), len(self.X_blobs))
        self.assertEqual(len(np.unique(labels)), 4)
        self.assertTrue(hasattr(mbk, 'cluster_centers_'))
        self.assertTrue(hasattr(mbk, 'inertia_'))
    
    def test_enhanced_affinity_propagation(self):
        """Test Enhanced Affinity Propagation"""
        eap = EnhancedAffinityPropagation(adaptive_damping=True, random_state=42)
        labels = eap.fit_predict(self.X_blobs[:100])  # Smaller dataset for AP
        
        self.assertEqual(len(labels), 100)
        self.assertTrue(len(np.unique(labels)) >= 2)  # Should find at least 2 clusters
        self.assertTrue(hasattr(eap, 'cluster_centers_'))


class TestAdaptiveOptimization(unittest.TestCase):
    """Test adaptive parameter optimization"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = make_blobs(n_samples=200, centers=3, n_features=4, 
                                        random_state=42, cluster_std=1.5)
    
    def test_bayesian_cluster_optimizer(self):
        """Test Bayesian optimization for clustering"""
        from sklearn.cluster import KMeans
        
        # Define parameter space
        param_space = {
            'n_clusters': {'type': 'int', 'range': (2, 6)},
            'max_iter': {'type': 'int', 'range': (50, 300)},
            'tol': {'type': 'float', 'range': (1e-6, 1e-2)}
        }
        
        optimizer = BayesianClusterOptimizer(n_initial_points=3, n_optimization_steps=5, 
                                           random_state=42)
        
        best_params, best_score = optimizer.optimize(
            self.X, KMeans, param_space, verbose=False
        )
        
        self.assertIsInstance(best_params, dict)
        self.assertIn('n_clusters', best_params)
        self.assertIsInstance(best_score, (int, float))
        self.assertTrue(len(optimizer.optimization_history_) > 0)
    
    def test_adaptive_parameter_recommender(self):
        """Test adaptive parameter recommendation"""
        recommender = AdaptiveParameterRecommender(random_state=42)
        
        # Test K-means recommendations
        kmeans_recs = recommender.recommend_parameters(
            self.X, 'K-Means', n_recommendations=2
        )
        
        self.assertIsInstance(kmeans_recs, list)
        self.assertEqual(len(kmeans_recs), 2)
        
        for rec in kmeans_recs:
            self.assertIn('n_clusters', rec)
            self.assertIn('confidence', rec)
            self.assertIn('reasoning', rec)
        
        # Test DBSCAN recommendations
        dbscan_recs = recommender.recommend_parameters(
            self.X, 'DBSCAN', n_recommendations=1
        )
        
        self.assertIsInstance(dbscan_recs, list)
        self.assertTrue(len(dbscan_recs) >= 1)
        self.assertIn('eps', dbscan_recs[0])
        self.assertIn('min_samples', dbscan_recs[0])
    
    def test_advanced_parameter_optimizer(self):
        """Test advanced parameter optimizer integration"""
        from sklearn.cluster import KMeans
        
        param_space = {
            'n_clusters': {'type': 'int', 'range': (2, 5)}
        }
        
        optimizer = AdvancedParameterOptimizer(random_state=42)
        
        results = optimizer.optimize_algorithm(
            self.X, 'K-Means', KMeans, param_space,
            optimization_method='adaptive', 
            n_recommendations=2,
            bayesian_steps=3,
            verbose=False
        )
        
        self.assertIn('adaptive_results', results)
        self.assertIn('best_overall_params', results)
        self.assertIn('best_overall_score', results)


class TestEvaluationMetrics(unittest.TestCase):
    """Test comprehensive evaluation metrics"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = make_blobs(n_samples=200, centers=3, n_features=4, 
                                        random_state=42)
        # Create clustering labels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.labels = kmeans.fit_predict(self.X)
    
    def test_comprehensive_evaluator(self):
        """Test comprehensive clustering evaluator"""
        evaluator = ComprehensiveClusteringEvaluator()
        
        # Test internal validation
        results = evaluator.evaluate_clustering(
            self.X, self.labels, algorithm_name='K-Means',
            include_categories=['internal']
        )
        
        self.assertIn('metrics', results)
        self.assertIn('internal', results['metrics'])
        internal_metrics = results['metrics']['internal']
        
        self.assertIn('silhouette_score', internal_metrics)
        self.assertIn('calinski_harabasz_index', internal_metrics)
        self.assertIn('davies_bouldin_index', internal_metrics)
        
        # Test with ground truth
        results_external = evaluator.evaluate_clustering(
            self.X, self.labels, ground_truth=self.y_true,
            include_categories=['internal', 'external']
        )
        
        self.assertIn('external', results_external['metrics'])
        external_metrics = results_external['metrics']['external']
        
        self.assertIn('adjusted_rand_score', external_metrics)
        self.assertIn('normalized_mutual_info', external_metrics)
        
        # Test overall score calculation
        self.assertIn('overall_score', results)
        self.assertIsInstance(results['overall_score'], (int, float))
    
    def test_evaluation_comparison(self):
        """Test comparison of multiple clustering results"""
        evaluator = ComprehensiveClusteringEvaluator()
        
        # Create multiple clustering results
        from sklearn.cluster import KMeans, DBSCAN
        
        kmeans_labels = KMeans(n_clusters=3, random_state=42).fit_predict(self.X)
        dbscan_labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(self.X)
        
        results_kmeans = evaluator.evaluate_clustering(
            self.X, kmeans_labels, algorithm_name='K-Means'
        )
        results_dbscan = evaluator.evaluate_clustering(
            self.X, dbscan_labels, algorithm_name='DBSCAN'
        )
        
        # Compare results
        comparison = evaluator.compare_clusterings([results_kmeans, results_dbscan])
        
        self.assertIn('ranking', comparison)
        self.assertIn('best_algorithm', comparison)
        self.assertIn('detailed_comparison', comparison)
        
        self.assertEqual(len(comparison['ranking']), 2)


class TestAlgorithmSelection(unittest.TestCase):
    """Test intelligent algorithm selection"""
    
    def setUp(self):
        """Set up test data"""
        self.X_simple, _ = make_blobs(n_samples=300, centers=4, random_state=42)
        self.X_complex, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    def test_data_characterization_engine(self):
        """Test data characterization"""
        engine = DataCharacterizationEngine()
        
        characteristics = engine.analyze_dataset(self.X_simple)
        
        # Check basic statistics
        self.assertIn('basic_stats', characteristics)
        basic_stats = characteristics['basic_stats']
        self.assertEqual(basic_stats['n_samples'], 300)
        self.assertEqual(basic_stats['n_features'], 2)
        
        # Check dimensionality analysis
        self.assertIn('dimensionality', characteristics)
        
        # Check structure analysis
        self.assertIn('structure', characteristics)
        structure = characteristics['structure']
        self.assertIn('hopkins_statistic', structure)
        
        # Check noise estimation
        self.assertIn('noise', characteristics)
    
    def test_algorithm_selector(self):
        """Test algorithm selection logic"""
        # Characterize data
        engine = DataCharacterizationEngine()
        characteristics = engine.analyze_dataset(self.X_simple)
        
        # Select algorithms
        selector = AlgorithmSelector()
        recommendations = selector.select_algorithms(
            characteristics, n_recommendations=3
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) <= 3)
        
        for rec in recommendations:
            self.assertIn('algorithm', rec)
            self.assertIn('score', rec)
            self.assertIn('confidence', rec)
            self.assertIn('suggested_parameters', rec)
    
    def test_intelligent_algorithm_selector(self):
        """Test complete intelligent selection pipeline"""
        selector = IntelligentAlgorithmSelector()
        
        # Test with simple data
        results = selector.recommend_algorithms(
            self.X_simple, n_recommendations=2, include_analysis=True
        )
        
        self.assertIn('recommendations', results)
        self.assertIn('top_choice', results)
        self.assertIn('data_characteristics', results)
        
        recommendations = results['recommendations']
        self.assertTrue(len(recommendations) <= 2)
        
        if recommendations:
            top_choice = recommendations[0]
            self.assertIn('algorithm', top_choice)
            self.assertIn('suggested_parameters', top_choice)


class TestInterpretability(unittest.TestCase):
    """Test clustering interpretability features"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = make_blobs(n_samples=200, centers=3, n_features=4, 
                                        random_state=42)
        from sklearn.cluster import KMeans
        self.labels = KMeans(n_clusters=3, random_state=42).fit_predict(self.X)
        self.feature_names = [f'feature_{i}' for i in range(4)]
    
    def test_cluster_profiler(self):
        """Test cluster profiling"""
        profiler = ClusterProfiler()
        
        profiles = profiler.create_cluster_profiles(
            self.X, self.labels, self.feature_names
        )
        
        self.assertIsInstance(profiles, dict)
        self.assertTrue(len(profiles) == 3)  # 3 clusters
        
        for cluster_name, profile in profiles.items():
            self.assertIn('basic_info', profile)
            self.assertIn('statistical_summary', profile)
            self.assertIn('feature_characteristics', profile)
            
            # Check basic info
            basic_info = profile['basic_info']
            self.assertIn('size', basic_info)
            self.assertIn('percentage', basic_info)
            
            # Check statistical summary
            statistical_summary = profile['statistical_summary']
            for feature_name in self.feature_names:
                self.assertIn(feature_name, statistical_summary)
                feature_stats = statistical_summary[feature_name]
                self.assertIn('mean', feature_stats)
                self.assertIn('std', feature_stats)
    
    def test_cluster_explainer(self):
        """Test cluster explanation"""
        explainer = ClusterExplainer()
        
        explanations = explainer.explain_clustering(
            self.X, self.labels, self.feature_names, method='decision_tree'
        )
        
        self.assertIn('global_feature_importance', explanations)
        self.assertIn('cluster_rules', explanations)
        self.assertIn('feature_distributions', explanations)
        
        # Check global importance
        global_importance = explanations['global_feature_importance']
        if 'feature_importance' in global_importance:
            feature_importance = global_importance['feature_importance']
            self.assertEqual(len(feature_importance), len(self.feature_names))
        
        # Check cluster rules
        cluster_rules = explanations['cluster_rules']
        self.assertTrue(len(cluster_rules) >= 1)  # At least one cluster should have rules
    
    def test_comprehensive_interpreter(self):
        """Test comprehensive clustering interpretation"""
        interpreter = ComprehensiveClusteringInterpreter()
        
        results = interpreter.interpret_clustering(
            self.X, self.labels, self.feature_names,
            include_visualizations=False,  # Skip visualizations for speed
            explanation_method='decision_tree'
        )
        
        self.assertIn('summary', results)
        self.assertIn('cluster_profiles', results)
        self.assertIn('explanations', results)
        self.assertIn('natural_language_summary', results)
        
        # Check summary
        summary = results['summary']
        self.assertEqual(summary['n_samples'], 200)
        self.assertEqual(summary['n_features'], 4)
        self.assertEqual(summary['n_clusters'], 3)


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features"""
    
    def setUp(self):
        """Set up test data"""
        # Create larger dataset for performance testing
        self.X_large, _ = make_blobs(n_samples=5000, centers=4, n_features=10, 
                                    random_state=42)
        self.X_small, _ = make_blobs(n_samples=200, centers=3, n_features=5, 
                                    random_state=42)
    
    def test_memory_efficient_scaler(self):
        """Test memory-efficient scaling"""
        scaler = MemoryEfficientScaler(chunk_size=1000)
        
        X_scaled = scaler.fit_transform(self.X_large)
        
        self.assertEqual(X_scaled.shape, self.X_large.shape)
        
        # Check that data is approximately standardized
        np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)
    
    def test_streaming_kmeans(self):
        """Test streaming K-means"""
        streaming_kmeans = StreamingKMeans(n_clusters=4, batch_size=1000, random_state=42)
        
        labels = streaming_kmeans.fit_predict(self.X_large)
        
        self.assertEqual(len(labels), len(self.X_large))
        self.assertEqual(len(np.unique(labels)), 4)
        self.assertTrue(hasattr(streaming_kmeans, 'cluster_centers_'))
        
        # Test partial fit
        streaming_kmeans2 = StreamingKMeans(n_clusters=4, random_state=42)
        
        # Fit in batches
        batch_size = 1000
        for i in range(0, len(self.X_large), batch_size):
            batch = self.X_large[i:i+batch_size]
            streaming_kmeans2.partial_fit(batch)
        
        self.assertTrue(hasattr(streaming_kmeans2, 'cluster_centers_'))
    
    def test_approximate_dbscan(self):
        """Test approximate DBSCAN"""
        approx_dbscan = ApproximateDBSCAN(
            eps=1.0, min_samples=10, sample_fraction=0.2, random_state=42
        )
        
        labels = approx_dbscan.fit_predict(self.X_large)
        
        self.assertEqual(len(labels), len(self.X_large))
        # Should find at least some clusters (not all noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.assertTrue(n_clusters >= 1)
    
    def test_scalable_clustering_pipeline(self):
        """Test scalable clustering pipeline"""
        pipeline = ScalableClusteringPipeline(memory_limit_gb=2.0, n_jobs=1)
        
        # Test with automatic algorithm selection
        labels, metadata = pipeline.fit_predict(self.X_large, algorithm='auto')
        
        self.assertEqual(len(labels), len(self.X_large))
        self.assertIn('processing_time', metadata)
        self.assertIn('data_characteristics', metadata)
        self.assertIn('clustering_strategy', metadata)
        
        # Test memory recommendations
        recommendations = pipeline.get_memory_recommendations(self.X_large)
        
        self.assertIn('data_characteristics', recommendations)
        self.assertIn('recommendations', recommendations)


class TestIntegration(unittest.TestCase):
    """Test integration between different components"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y_true = make_blobs(n_samples=300, centers=4, n_features=6, 
                                        random_state=42)
        self.feature_names = [f'feature_{i}' for i in range(6)]
    
    def test_full_pipeline_integration(self):
        """Test full pipeline from selection to interpretation"""
        # Step 1: Intelligent algorithm selection
        selector = IntelligentAlgorithmSelector()
        selection_results = selector.recommend_algorithms(
            self.X, n_recommendations=1, include_analysis=False
        )
        
        self.assertIn('recommendations', selection_results)
        self.assertTrue(len(selection_results['recommendations']) >= 1)
        
        top_choice = selection_results['recommendations'][0]
        algorithm_name = top_choice['algorithm']
        suggested_params = top_choice['suggested_parameters']
        
        # Step 2: Apply selected algorithm
        if algorithm_name == 'K-Means':
            from sklearn.cluster import KMeans
            clusterer = KMeans(**suggested_params)
        elif algorithm_name == 'DBSCAN':
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(**suggested_params)
        else:
            # Fallback
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=4)
        
        labels = clusterer.fit_predict(self.X)
        
        # Step 3: Comprehensive evaluation
        evaluator = ComprehensiveClusteringEvaluator()
        evaluation_results = evaluator.evaluate_clustering(
            self.X, labels, ground_truth=self.y_true,
            algorithm_name=algorithm_name,
            include_categories=['internal', 'external']
        )
        
        self.assertIn('overall_score', evaluation_results)
        
        # Step 4: Interpretation
        interpreter = ComprehensiveClusteringInterpreter()
        interpretation_results = interpreter.interpret_clustering(
            self.X, labels, self.feature_names,
            include_visualizations=False
        )
        
        self.assertIn('cluster_profiles', interpretation_results)
        self.assertIn('explanations', interpretation_results)
    
    def test_parameter_optimization_integration(self):
        """Test parameter optimization with evaluation"""
        from sklearn.cluster import KMeans
        
        # Define parameter space
        param_space = {
            'n_clusters': {'type': 'int', 'range': (3, 6)},
            'max_iter': {'type': 'int', 'range': (50, 200)}
        }
        
        # Optimize parameters
        optimizer = AdvancedParameterOptimizer(random_state=42)
        optimization_results = optimizer.optimize_algorithm(
            self.X, 'K-Means', KMeans, param_space,
            optimization_method='adaptive',
            verbose=False
        )
        
        self.assertIn('best_overall_params', optimization_results)
        
        # Use optimized parameters for clustering
        best_params = optimization_results['best_overall_params']
        if best_params:
            clusterer = KMeans(**best_params)
            labels = clusterer.fit_predict(self.X)
            
            # Evaluate optimized clustering
            evaluator = ComprehensiveClusteringEvaluator()
            evaluation_results = evaluator.evaluate_clustering(
                self.X, labels, algorithm_name='Optimized K-Means'
            )
            
            self.assertIn('overall_score', evaluation_results)


def run_all_tests():
    """Run all test suites"""
    test_suites = [
        TestAdvancedAlgorithms,
        TestAdaptiveOptimization,
        TestEvaluationMetrics,
        TestAlgorithmSelection,
        TestInterpretability,
        TestPerformanceOptimization,
        TestIntegration
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_suite_class in test_suites:
        print(f"\n{'='*60}")
        print(f"Running {test_suite_class.__name__}")
        print(f"{'='*60}")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
    
    print(f"\n{'='*60}")
    print(f"OVERALL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    
    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"❌ {total_failures + total_errors} tests failed")
        return False


if __name__ == '__main__':
    print("COMPREHENSIVE CLUSTERING MODULE TEST SUITE")
    print("=" * 80)
    print("Testing all enhanced clustering features...")
    print("This may take several minutes due to comprehensive testing.")
    print("=" * 80)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All enhanced clustering features are working correctly!")
        print("The clustering module is ready for production use.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("Fix any issues before using the enhanced features.")
    
    print("\nTest suite completed.")