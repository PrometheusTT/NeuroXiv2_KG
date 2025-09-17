"""
Enhanced tools for more powerful agent capabilities.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from .neo4j_exec import Neo4jExec
from .schema_cache import SchemaCache
from .tools_stats import basic_stats, mismatch_index

logger = logging.getLogger(__name__)


class EnhancedAnalysisTools:
    """Advanced analysis tools for knowledge graph exploration."""

    def __init__(self, db: Neo4jExec, schema: SchemaCache):
        self.db = db
        self.schema = schema
        self._cache = {}

    def compute_graph_metrics(self, node_type: str = "Region",
                            relationship_type: str = "PROJECT_TO") -> Dict[str, Any]:
        """Compute comprehensive graph metrics for network analysis."""
        cache_key = f"graph_metrics_{node_type}_{relationship_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get graph structure
        query = f"""
        MATCH (n:{node_type})-[r:{relationship_type}]->(m:{node_type})
        RETURN n.acronym AS source, m.acronym AS target,
               coalesce(r.weight, r.strength, 1.0) AS weight
        """
        result = self.db.run(query)

        if not result["success"] or not result["data"]:
            return {"error": "No graph data found"}

        # Build NetworkX graph
        G = nx.DiGraph()
        for row in result["data"]:
            G.add_edge(row["source"], row["target"], weight=row["weight"])

        metrics = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_weakly_connected(G),
            "avg_clustering": nx.average_clustering(G.to_undirected()),
        }

        # Centrality measures
        try:
            metrics["degree_centrality"] = dict(nx.degree_centrality(G))
            metrics["betweenness_centrality"] = dict(nx.betweenness_centrality(G, k=min(100, len(G))))
            metrics["pagerank"] = dict(nx.pagerank(G, max_iter=100))
            metrics["eigenvector_centrality"] = dict(nx.eigenvector_centrality(G, max_iter=100))
        except Exception as e:
            logger.warning(f"Centrality computation failed: {e}")
            metrics["centrality_error"] = str(e)

        # Community detection
        try:
            undirected_G = G.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected_G)
            metrics["num_communities"] = len(communities)
            metrics["modularity"] = nx.community.modularity(undirected_G, communities)
            metrics["communities"] = [list(c) for c in communities]
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")

        self._cache[cache_key] = metrics
        return metrics

    def find_shortest_paths(self, source: str, target: str,
                          node_type: str = "Region",
                          relationship_type: str = "PROJECT_TO") -> Dict[str, Any]:
        """Find and analyze shortest paths between nodes."""
        query = f"""
        MATCH path = shortestPath((s:{node_type} {{acronym: $source}})-[:{relationship_type}*..10]->(t:{node_type} {{acronym: $target}}))
        RETURN path, length(path) as path_length
        """
        result = self.db.run(query, {"source": source, "target": target})

        if not result["success"]:
            return {"error": result.get("error", "Path query failed")}

        paths = []
        for row in result["data"]:
            path_info = {
                "length": row["path_length"],
                "path": row["path"]  # This would need proper parsing in practice
            }
            paths.append(path_info)

        return {
            "source": source,
            "target": target,
            "paths_found": len(paths),
            "shortest_length": min([p["length"] for p in paths]) if paths else None,
            "paths": paths[:5]  # Return top 5 paths
        }

    def analyze_node_neighborhoods(self, node_id: str, node_type: str = "Region",
                                 max_depth: int = 2) -> Dict[str, Any]:
        """Analyze the neighborhood structure around a specific node."""
        query = f"""
        MATCH (center:{node_type} {{acronym: $node_id}})
        OPTIONAL MATCH (center)-[r1]-(neighbor1)
        OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2)
        WHERE neighbor2 IS NULL OR id(neighbor2) <> id(center)
        RETURN center, neighbor1, neighbor2, r1, r2
        LIMIT 1000
        """
        result = self.db.run(query, {"node_id": node_id})

        if not result["success"]:
            return {"error": "Neighborhood query failed"}

        neighbors_1hop = set()
        neighbors_2hop = set()
        edge_types = Counter()

        for row in result["data"]:
            if row.get("neighbor1"):
                neighbors_1hop.add(row["neighbor1"].get("acronym", "unknown"))
                if row.get("r1"):
                    edge_types[type(row["r1"]).__name__] += 1
            if row.get("neighbor2"):
                neighbors_2hop.add(row["neighbor2"].get("acronym", "unknown"))

        return {
            "center_node": node_id,
            "neighbors_1hop": len(neighbors_1hop),
            "neighbors_2hop": len(neighbors_2hop),
            "edge_type_distribution": dict(edge_types),
            "neighbor_list_1hop": list(neighbors_1hop)[:20],
            "neighbor_list_2hop": list(neighbors_2hop)[:20]
        }

    def cluster_analysis(self, feature_query: str, n_clusters: int = 5) -> Dict[str, Any]:
        """Perform clustering analysis on node features."""
        result = self.db.run(feature_query)

        if not result["success"] or not result["data"]:
            return {"error": "Feature query failed or returned no data"}

        # Extract features and labels
        data = result["data"]
        features = []
        labels = []

        for row in data:
            # Assume numeric features (adjust based on your data structure)
            feature_values = []
            label = None

            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    feature_values.append(float(value))
                elif label is None:  # Use first string value as label
                    label = str(value)

            if feature_values and label:
                features.append(feature_values)
                labels.append(label)

        if len(features) < n_clusters:
            return {"error": f"Not enough data points ({len(features)}) for {n_clusters} clusters"}

        # Standardize features
        features_array = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Analyze clusters
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(cluster_labels):
            clusters[cluster_id].append(labels[i])

        # Compute cluster statistics
        cluster_stats = {}
        for cluster_id, members in clusters.items():
            cluster_features = features_scaled[cluster_labels == cluster_id]
            cluster_stats[cluster_id] = {
                "size": len(members),
                "members": members[:10],  # Top 10 members
                "centroid": kmeans.cluster_centers_[cluster_id].tolist(),
                "inertia": float(np.sum((cluster_features - kmeans.cluster_centers_[cluster_id]) ** 2))
            }

        return {
            "n_clusters": n_clusters,
            "total_points": len(features),
            "total_inertia": float(kmeans.inertia_),
            "cluster_stats": cluster_stats,
            "silhouette_score": self._compute_silhouette_score(features_scaled, cluster_labels)
        }

    def _compute_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(features, labels))
        except Exception:
            return 0.0

    def statistical_analysis(self, data_query: str,
                           group_by_column: str = None) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on query results."""
        result = self.db.run(data_query)

        if not result["success"] or not result["data"]:
            return {"error": "Data query failed or returned no data"}

        data = result["data"]
        numeric_columns = []
        categorical_columns = []

        # Identify column types
        if data:
            for key, value in data[0].items():
                if isinstance(value, (int, float)) and value is not None:
                    numeric_columns.append(key)
                else:
                    categorical_columns.append(key)

        analysis = {
            "total_records": len(data),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns
        }

        # Analyze numeric columns
        for col in numeric_columns:
            values = [float(row[col]) for row in data if row[col] is not None]
            if values:
                analysis[f"{col}_stats"] = basic_stats(values)
                analysis[f"{col}_distribution"] = {
                    "histogram": np.histogram(values, bins=10)[0].tolist(),
                    "histogram_bins": np.histogram(values, bins=10)[1].tolist()
                }

        # Analyze categorical columns
        for col in categorical_columns:
            values = [str(row[col]) for row in data if row[col] is not None]
            if values:
                counter = Counter(values)
                analysis[f"{col}_distribution"] = dict(counter.most_common(20))

        # Group-by analysis if specified
        if group_by_column and group_by_column in categorical_columns:
            groups = defaultdict(list)
            for row in data:
                group_val = str(row.get(group_by_column, "unknown"))
                groups[group_val].append(row)

            group_analysis = {}
            for group_name, group_data in groups.items():
                group_stats = {"count": len(group_data)}
                for col in numeric_columns:
                    values = [float(r[col]) for r in group_data if r[col] is not None]
                    if values:
                        group_stats[f"{col}_stats"] = basic_stats(values)
                group_analysis[group_name] = group_stats

            analysis["group_analysis"] = group_analysis

        return analysis

    def correlation_analysis(self, data_query: str) -> Dict[str, Any]:
        """Compute correlation matrix for numeric features."""
        result = self.db.run(data_query)

        if not result["success"] or not result["data"]:
            return {"error": "Data query failed"}

        data = result["data"]
        numeric_data = {}

        # Extract numeric columns
        for row in data:
            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    if key not in numeric_data:
                        numeric_data[key] = []
                    numeric_data[key].append(float(value))

        if len(numeric_data) < 2:
            return {"error": "Need at least 2 numeric columns for correlation"}

        # Ensure all columns have same length
        min_length = min(len(values) for values in numeric_data.values())
        for key in numeric_data:
            numeric_data[key] = numeric_data[key][:min_length]

        # Compute correlation matrix
        columns = list(numeric_data.keys())
        correlation_matrix = np.corrcoef([numeric_data[col] for col in columns])

        # Convert to dictionary format
        correlations = {}
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                correlations[f"{col1}_vs_{col2}"] = float(correlation_matrix[i, j])

        return {
            "columns": columns,
            "correlations": correlations,
            "correlation_matrix": correlation_matrix.tolist()
        }

    def pattern_mining(self, pattern_query: str,
                      min_support: float = 0.1) -> Dict[str, Any]:
        """Mine frequent patterns in the data."""
        result = self.db.run(pattern_query)

        if not result["success"] or not result["data"]:
            return {"error": "Pattern query failed"}

        data = result["data"]

        # Extract patterns (assuming data contains categorical features)
        transactions = []
        for row in data:
            transaction = []
            for key, value in row.items():
                if value is not None:
                    transaction.append(f"{key}:{value}")
            transactions.append(transaction)

        # Count item frequencies
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        total_transactions = len(transactions)
        min_count = int(min_support * total_transactions)

        # Find frequent items
        frequent_items = {item: count for item, count in item_counts.items()
                         if count >= min_count}

        # Find frequent pairs (simplified association rules)
        pair_counts = Counter()
        for transaction in transactions:
            frequent_in_transaction = [item for item in transaction if item in frequent_items]
            for i in range(len(frequent_in_transaction)):
                for j in range(i + 1, len(frequent_in_transaction)):
                    pair = tuple(sorted([frequent_in_transaction[i], frequent_in_transaction[j]]))
                    pair_counts[pair] += 1

        frequent_pairs = {pair: count for pair, count in pair_counts.items()
                         if count >= min_count}

        return {
            "total_transactions": total_transactions,
            "min_support_count": min_count,
            "frequent_items": dict(sorted(frequent_items.items(), key=lambda x: x[1], reverse=True)[:20]),
            "frequent_pairs": dict(sorted(frequent_pairs.items(), key=lambda x: x[1], reverse=True)[:20]),
            "num_frequent_items": len(frequent_items),
            "num_frequent_pairs": len(frequent_pairs)
        }

    def anomaly_detection(self, data_query: str,
                         contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies in the data using statistical methods."""
        result = self.db.run(data_query)

        if not result["success"] or not result["data"]:
            return {"error": "Data query failed"}

        data = result["data"]

        # Extract numeric features and identifiers
        features = []
        identifiers = []

        for row in data:
            feature_values = []
            identifier = None

            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    feature_values.append(float(value))
                elif identifier is None:
                    identifier = str(value)

            if feature_values and identifier:
                features.append(feature_values)
                identifiers.append(identifier)

        if len(features) < 10:
            return {"error": "Need at least 10 data points for anomaly detection"}

        features_array = np.array(features)

        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(features_array, axis=0))
        z_score_threshold = 3.0
        z_anomalies = np.any(z_scores > z_score_threshold, axis=1)

        # IQR based anomaly detection
        Q1 = np.percentile(features_array, 25, axis=0)
        Q3 = np.percentile(features_array, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        iqr_anomalies = np.any((features_array < lower_bound) | (features_array > upper_bound), axis=1)

        # Combine results
        anomaly_scores = z_scores.max(axis=1)
        anomaly_indices = np.where(z_anomalies | iqr_anomalies)[0]

        anomalies = []
        for idx in anomaly_indices[:20]:  # Top 20 anomalies
            anomalies.append({
                "identifier": identifiers[idx],
                "z_score": float(anomaly_scores[idx]),
                "features": features[idx]
            })

        return {
            "total_points": len(features),
            "z_score_anomalies": int(np.sum(z_anomalies)),
            "iqr_anomalies": int(np.sum(iqr_anomalies)),
            "combined_anomalies": len(anomaly_indices),
            "anomaly_rate": len(anomaly_indices) / len(features),
            "top_anomalies": anomalies
        }


class VisualizationTools:
    """Tools for generating visualizations and plots."""

    @staticmethod
    def create_distribution_plot(data: List[float], title: str = "Distribution") -> str:
        """Create a histogram and return as base64 encoded image."""
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)

            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()

            return image_base64
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return ""

    @staticmethod
    def create_correlation_heatmap(correlation_matrix: List[List[float]],
                                 labels: List[str], title: str = "Correlation Matrix") -> str:
        """Create a correlation heatmap and return as base64 encoded image."""
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       xticklabels=labels, yticklabels=labels, fmt='.2f')
            plt.title(title)
            plt.tight_layout()

            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()

            return image_base64
        except Exception as e:
            logger.error(f"Heatmap creation failed: {e}")
            return ""