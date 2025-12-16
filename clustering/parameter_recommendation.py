"""
Intelligent Parameter Recommendation System
Provides data-driven parameter suggestions for clustering algorithms
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class ParameterRecommendationEngine:
    """
    Intelligent parameter recommendation engine for clustering algorithms
    """
    
    def __init__(self, X, sample_size=1000):
        """
        Initialize the recommendation engine
        
        Args:
            X: Feature matrix
            sample_size: Maximum sample size for analysis
        """
        self.X = X
        if len(X) > sample_size:
            # Random sampling for large datasets
            indices = np.random.choice(len(X), sample_size, replace=False)
            self.X_sample = X[indices]
        else:
            self.X_sample = X.copy()
            
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_sample)
        
    def recommend_kmeans_k(self, k_range=None, methods=['elbow', 'silhouette']):
        """
        Recommend optimal K for K-Means clustering
        
        Args:
            k_range: Range of K values to test
            methods: Methods to use for recommendation
            
        Returns:
            dict with recommendations
        """
        if k_range is None:
            max_k = min(10, len(self.X_scaled) // 10)
            k_range = range(2, max(3, max_k + 1))
            
        recommendations = {
            'k_range': list(k_range),
            'methods_used': methods,
            'recommendations': {}
        }
        
        if 'elbow' in methods:
            recommendations['recommendations']['elbow'] = self._elbow_method(k_range)
            
        if 'silhouette' in methods:
            recommendations['recommendations']['silhouette'] = self._silhouette_method(k_range)
            
        # Combine recommendations
        recommendations['final_recommendation'] = self._combine_k_recommendations(recommendations['recommendations'])
        
        return recommendations
        
    def _elbow_method(self, k_range):
        """Elbow method for K selection"""
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
        # Find elbow point using simple derivative method
        differences = np.diff(inertias)
        second_differences = np.diff(differences)
        
        if len(second_differences) > 0:
            elbow_index = np.argmax(second_differences) + 2  # +2 because of double diff
            elbow_k = list(k_range)[min(elbow_index, len(k_range)-1)]
        else:
            elbow_k = list(k_range)[len(k_range)//2]  # Fallback
            
        return {
            'recommended_k': elbow_k,
            'inertias': inertias,
            'confidence': self._calculate_elbow_confidence(inertias)
        }
        
    def _silhouette_method(self, k_range):
        """Silhouette analysis for K selection"""
        silhouette_scores = []
        for k in k_range:
            if k < len(self.X_scaled):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.X_scaled)
                score = silhouette_score(self.X_scaled, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
                
        best_k_index = np.argmax(silhouette_scores)
        best_k = list(k_range)[best_k_index]
        
        return {
            'recommended_k': best_k,
            'silhouette_scores': silhouette_scores,
            'best_score': silhouette_scores[best_k_index],
            'confidence': min(silhouette_scores[best_k_index], 1.0)
        }
        
    def _combine_k_recommendations(self, recommendations):
        """Combine multiple K recommendations"""
        if 'elbow' in recommendations and 'silhouette' in recommendations:
            elbow_k = recommendations['elbow']['recommended_k']
            sil_k = recommendations['silhouette']['recommended_k']
            elbow_conf = recommendations['elbow']['confidence']
            sil_conf = recommendations['silhouette']['confidence']
            
            # Weight recommendations by confidence
            if abs(elbow_k - sil_k) <= 1:  # Close agreement
                final_k = int((elbow_k * elbow_conf + sil_k * sil_conf) / (elbow_conf + sil_conf))
                confidence = (elbow_conf + sil_conf) / 2
                agreement = 'high'
            else:  # Disagreement
                if sil_conf > elbow_conf:
                    final_k = sil_k
                    confidence = sil_conf * 0.7  # Reduce confidence due to disagreement
                else:
                    final_k = elbow_k
                    confidence = elbow_conf * 0.7
                agreement = 'low'
                
            return {
                'recommended_k': final_k,
                'confidence': confidence,
                'agreement': agreement,
                'explanation': f"Elbow method suggests k={elbow_k}, Silhouette suggests k={sil_k}"
            }
        elif 'elbow' in recommendations:
            return recommendations['elbow']
        elif 'silhouette' in recommendations:
            return recommendations['silhouette']
        else:
            return {'recommended_k': 3, 'confidence': 0.5, 'explanation': 'Default recommendation'}
            
    def recommend_dbscan_eps(self, k=4, percentile=90):
        """
        Recommend eps parameter for DBSCAN using k-distance plot
        
        Args:
            k: Number of nearest neighbors to consider
            percentile: Percentile of k-distances to use as eps
            
        Returns:
            dict with eps recommendation
        """
        if len(self.X_scaled) < k:
            return {
                'recommended_eps': 0.5,
                'confidence': 0.3,
                'explanation': 'Insufficient data for k-distance analysis'
            }
            
        # Compute k-distances
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X_scaled)
        distances, indices = neighbors.kneighbors(self.X_scaled)
        
        # Sort k-distances (distance to kth nearest neighbor)
        k_distances = np.sort(distances[:, k-1])
        
        # Find knee point using percentile method
        knee_eps = np.percentile(k_distances, percentile)
        
        # Calculate confidence based on the clarity of the knee
        knee_clarity = self._calculate_knee_clarity(k_distances, knee_eps)
        
        return {
            'recommended_eps': knee_eps,
            'confidence': knee_clarity,
            'k_used': k,
            'percentile_used': percentile,
            'k_distances': k_distances,
            'explanation': f'Based on {percentile}th percentile of {k}-distances'
        }
        
    def recommend_dbscan_min_samples(self, dimensionality=None):
        """
        Recommend min_samples for DBSCAN
        
        Args:
            dimensionality: Number of dimensions (if None, uses actual dimensionality)
            
        Returns:
            dict with min_samples recommendation
        """
        if dimensionality is None:
            dimensionality = self.X_scaled.shape[1]
            
        # Common heuristic: min_samples >= dimensionality + 1
        recommended_min_samples = max(4, dimensionality + 1)
        
        # Adjust based on dataset size
        if len(self.X_scaled) < 100:
            recommended_min_samples = max(3, recommended_min_samples - 1)
        elif len(self.X_scaled) > 1000:
            recommended_min_samples = min(10, recommended_min_samples + 2)
            
        confidence = 0.8 if 5 <= recommended_min_samples <= 8 else 0.6
        
        return {
            'recommended_min_samples': recommended_min_samples,
            'confidence': confidence,
            'dimensionality': dimensionality,
            'explanation': f'Based on dimensionality ({dimensionality}) + 1 heuristic'
        }
        
    def recommend_meanshift_bandwidth(self, quantile=0.2, n_samples=500):
        """
        Recommend bandwidth for Mean Shift clustering
        
        Args:
            quantile: Quantile for bandwidth estimation
            n_samples: Sample size for estimation
            
        Returns:
            dict with bandwidth recommendation  
        """
        sample_size = min(n_samples, len(self.X_scaled))
        
        try:
            bandwidth = estimate_bandwidth(
                self.X_scaled[:sample_size], 
                quantile=quantile, 
                n_samples=sample_size
            )
            
            if bandwidth <= 0:
                # Fallback to simple heuristic
                std_dev = np.std(self.X_scaled, axis=0).mean()
                bandwidth = std_dev * 0.5
                confidence = 0.4
                explanation = 'Fallback bandwidth based on standard deviation'
            else:
                confidence = 0.8
                explanation = f'Estimated using {quantile} quantile of pairwise distances'
                
        except Exception:
            # Ultimate fallback
            bandwidth = 1.0
            confidence = 0.3
            explanation = 'Default bandwidth due to estimation failure'
            
        return {
            'recommended_bandwidth': bandwidth,
            'confidence': confidence,
            'quantile_used': quantile,
            'sample_size': sample_size,
            'explanation': explanation
        }
        
    def get_data_characteristics(self):
        """
        Analyze data characteristics to suggest appropriate algorithms
        
        Returns:
            dict with data characteristics and algorithm suggestions
        """
        n_samples, n_features = self.X_scaled.shape
        
        # Compute basic statistics
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'density': self._estimate_density(),
            'noise_level': self._estimate_noise_level(),
            'dimensionality_ratio': n_samples / n_features if n_features > 0 else 0,
            'suggested_algorithms': []
        }
        
        # Algorithm suggestions based on characteristics
        if characteristics['density'] > 0.7 and n_samples < 5000:
            characteristics['suggested_algorithms'].extend(['K-Means', 'Gaussian Mixture'])
            
        if characteristics['noise_level'] > 0.3:
            characteristics['suggested_algorithms'].extend(['DBSCAN', 'OPTICS'])
            
        if n_samples > 10000:
            characteristics['suggested_algorithms'].extend(['BIRCH', 'Mean Shift'])
            
        if n_features > 10:
            characteristics['suggested_algorithms'].extend(['Spectral Clustering'])
            
        if not characteristics['suggested_algorithms']:
            characteristics['suggested_algorithms'] = ['K-Means', 'DBSCAN']
            
        return characteristics
        
    def _calculate_elbow_confidence(self, inertias):
        """Calculate confidence in elbow method recommendation"""
        if len(inertias) < 3:
            return 0.5
            
        # Calculate the sharpness of the elbow
        normalized_inertias = np.array(inertias) / max(inertias)
        differences = np.abs(np.diff(normalized_inertias))
        
        if len(differences) > 1:
            max_diff = max(differences)
            confidence = min(max_diff * 2, 1.0)  # Scale to 0-1
        else:
            confidence = 0.5
            
        return confidence
        
    def _calculate_knee_clarity(self, k_distances, knee_point):
        """Calculate how clear the knee point is in k-distance plot"""
        knee_index = np.searchsorted(k_distances, knee_point)
        
        if knee_index < len(k_distances) // 4 or knee_index > 3 * len(k_distances) // 4:
            return 0.3  # Knee is too early or too late
            
        # Calculate slope changes around knee
        before_slope = k_distances[knee_index] - k_distances[max(0, knee_index - 10)]
        after_slope = k_distances[min(len(k_distances)-1, knee_index + 10)] - k_distances[knee_index]
        
        if before_slope > 0 and after_slope > 0:
            slope_ratio = after_slope / before_slope
            confidence = min(slope_ratio / 3, 1.0)  # Higher ratio = clearer knee
        else:
            confidence = 0.5
            
        return confidence
        
    def _estimate_density(self):
        """Estimate data density (simplified)"""
        if len(self.X_scaled) < 10:
            return 0.5
            
        # Use k-nearest neighbors to estimate local density
        k = min(10, len(self.X_scaled) - 1)
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X_scaled)
        distances, _ = neighbors.kneighbors(self.X_scaled)
        
        avg_distance = np.mean(distances[:, -1])  # Distance to kth neighbor
        # Normalize to 0-1 scale (lower distance = higher density)
        max_possible_distance = np.sqrt(self.X_scaled.shape[1])  # Max distance in unit hypercube
        density = max(0, 1 - (avg_distance / max_possible_distance))
        
        return density
        
    def _estimate_noise_level(self):
        """Estimate noise level in data (simplified)"""
        if len(self.X_scaled) < 20:
            return 0.3
            
        # Use local outlier factor concept
        k = min(20, len(self.X_scaled) - 1)
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X_scaled)
        distances, _ = neighbors.kneighbors(self.X_scaled)
        
        # Points with unusually large distances to neighbors are likely noise
        mean_distances = np.mean(distances, axis=1)
        threshold = np.percentile(mean_distances, 80)
        
        noise_fraction = np.mean(mean_distances > threshold)
        return min(noise_fraction * 2, 1.0)  # Scale to reasonable range