from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx

def _vec(keys: Sequence[str], d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d.get(k, 0)) for k in keys], dtype=float)


def _dist(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    metric = (metric or "L1").upper()
    if metric == "L2":
        return float(np.linalg.norm(x - y))
    if metric in ("COS", "COSINE"):
        if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
            return 1.0
        return float(1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return float(np.sum(np.abs(x - y)))  # L1

def mismatch_index(
    morph_vec_a: Dict[str, float], morph_vec_b: Dict[str, float],
    subclass_vec_a: Dict[str, float], subclass_vec_b: Dict[str, float],
    metric: str = "L1"
) -> float:
    """
    | dist(morph_a, morph_b) - dist(subclass_a, subclass_b) |
    Vectors are aligned by keys with zeros filled for missing entries.
    """
    keys_m = sorted(set(morph_vec_a) | set(morph_vec_b))
    keys_s = sorted(set(subclass_vec_a) | set(subclass_vec_b))
    va, vb = _vec(keys_m, morph_vec_a), _vec(keys_m, morph_vec_b)
    sa, sb = _vec(keys_s, subclass_vec_a), _vec(keys_s, subclass_vec_b)
    return abs(_dist(va, vb, metric) - _dist(sa, sb, metric))


def basic_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
def detect_projection_patterns(projection_data: List[Dict]) -> Dict:
    """
    Detect projection patterns matching paper Figure 3
    """
    patterns = {
        "hub_regions": [],
        "projection_clusters": [],
        "specificity_scores": {},
        "hierarchical_levels": []
    }

    # Build projection matrix
    projections = defaultdict(lambda: defaultdict(float))
    regions = set()

    for row in projection_data:
        if all(k in row for k in ["source", "target", "weight"]):
            src, tgt, weight = row["source"], row["target"], float(row["weight"])
            projections[src][tgt] = weight
            regions.update([src, tgt])

    # Identify hub regions (high out-degree and in-degree)
    out_degree = {r: len(projections[r]) for r in regions}
    in_degree = {r: sum(1 for s in projections if r in projections[s]) for r in regions}

    # Hub score = geometric mean of in and out degree
    hub_scores = {}
    for r in regions:
        if out_degree[r] > 0 and in_degree[r] > 0:
            hub_scores[r] = np.sqrt(out_degree[r] * in_degree[r])

    # Top hubs
    top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    patterns["hub_regions"] = [
        {
            "region": hub,
            "score": score,
            "out_degree": out_degree[hub],
            "in_degree": in_degree[hub],
            "top_targets": sorted(
                [(t, w) for t, w in projections[hub].items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        for hub, score in top_hubs
    ]

    # Detect projection clusters using hierarchical clustering
    if len(regions) > 3:
        # Create projection similarity matrix
        region_list = sorted(regions)
        n = len(region_list)
        similarity_matrix = np.zeros((n, n))

        for i, r1 in enumerate(region_list):
            for j, r2 in enumerate(region_list):
                if i != j:
                    # Jaccard similarity of projection targets
                    targets1 = set(projections[r1].keys())
                    targets2 = set(projections[r2].keys())
                    if targets1 or targets2:
                        jaccard = len(targets1 & targets2) / len(targets1 | targets2)
                        similarity_matrix[i, j] = 1 - jaccard

        # Perform clustering
        if np.any(similarity_matrix):
            linkage_matrix = linkage(squareform(similarity_matrix), method='ward')
            clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

            for cluster_id in range(1, 4):
                cluster_regions = [region_list[i] for i, c in enumerate(clusters) if c == cluster_id]
                if cluster_regions:
                    patterns["projection_clusters"].append({
                        "cluster_id": cluster_id,
                        "regions": cluster_regions,
                        "size": len(cluster_regions)
                    })

    # Calculate specificity scores
    for region in regions:
        targets = projections[region]
        if targets:
            # Entropy-based specificity (lower = more specific)
            weights = np.array(list(targets.values()))
            weights = weights / weights.sum()
            entropy = -np.sum(weights * np.log(weights + 1e-10))
            specificity = 1 / (1 + entropy)
            patterns["specificity_scores"][region] = specificity

    return patterns


def compute_morphology_molecular_mismatch(
        morphology_data: Dict[str, Dict],
        molecular_data: Dict[str, Dict],
        metric: str = "L1"
) -> Dict:
    """
    Compute mismatch index as shown in paper Figure 4B
    """
    results = {
        "pairwise_mismatches": [],
        "mismatch_matrix": None,
        "top_mismatches": [],
        "statistical_summary": {}
    }

    common_regions = set(morphology_data.keys()) & set(molecular_data.keys())
    if len(common_regions) < 2:
        return results

    region_list = sorted(common_regions)
    n = len(region_list)
    mismatch_matrix = np.zeros((n, n))

    for i, r1 in enumerate(region_list):
        for j, r2 in enumerate(region_list):
            if i < j:
                # Compute morphological distance
                morph1 = morphology_data[r1]
                morph2 = morphology_data[r2]
                morph_dist = _compute_distance(morph1, morph2, metric)

                # Compute molecular distance
                mol1 = molecular_data[r1]
                mol2 = molecular_data[r2]
                mol_dist = _compute_distance(mol1, mol2, metric)

                # Mismatch index
                mismatch = abs(morph_dist - mol_dist)
                mismatch_matrix[i, j] = mismatch_matrix[j, i] = mismatch

                results["pairwise_mismatches"].append({
                    "region1": r1,
                    "region2": r2,
                    "morphological_distance": morph_dist,
                    "molecular_distance": mol_dist,
                    "mismatch_index": mismatch
                })

    results["mismatch_matrix"] = mismatch_matrix.tolist()

    # Top mismatches
    results["top_mismatches"] = sorted(
        results["pairwise_mismatches"],
        key=lambda x: x["mismatch_index"],
        reverse=True
    )[:10]

    # Statistical summary
    all_mismatches = [m["mismatch_index"] for m in results["pairwise_mismatches"]]
    if all_mismatches:
        results["statistical_summary"] = {
            "mean": np.mean(all_mismatches),
            "std": np.std(all_mismatches),
            "median": np.median(all_mismatches),
            "q25": np.percentile(all_mismatches, 25),
            "q75": np.percentile(all_mismatches, 75),
            "max": np.max(all_mismatches)
        }

    return results


def compute_network_metrics(edges: List[Tuple[str, str, float]]) -> Dict:
    """
    Compute comprehensive network metrics
    """
    G = nx.DiGraph()
    for src, tgt, weight in edges:
        G.add_edge(src, tgt, weight=weight)

    metrics = {
        "basic_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        },
        "centrality": {},
        "community_structure": {},
        "path_statistics": {}
    }

    # Centrality measures
    if G.number_of_nodes() > 0:
        metrics["centrality"] = {
            "betweenness": nx.betweenness_centrality(G),
            "eigenvector": nx.eigenvector_centrality(G, max_iter=100) if G.number_of_nodes() > 1 else {},
            "pagerank": nx.pagerank(G),
            "degree": dict(G.degree())
        }

        # Find top central nodes
        for centrality_type in ["betweenness", "eigenvector", "pagerank"]:
            if centrality_type in metrics["centrality"]:
                top_nodes = sorted(
                    metrics["centrality"][centrality_type].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                metrics["centrality"][f"top_{centrality_type}"] = top_nodes

    # Community detection (if graph is connected enough)
    if G.number_of_edges() > 10:
        try:
            communities = nx.community.louvain_communities(G.to_undirected())
            metrics["community_structure"] = {
                "num_communities": len(communities),
                "modularity": nx.community.modularity(G.to_undirected(), communities),
                "communities": [list(c) for c in communities[:5]]  # Top 5 communities
            }
        except:
            pass

    # Path statistics
    if nx.is_weakly_connected(G):
        metrics["path_statistics"] = {
            "average_shortest_path": nx.average_shortest_path_length(G),
            "diameter": nx.diameter(G.to_undirected())
        }

    return metrics


def identify_outliers(data: List[Dict], value_field: str = "value") -> List[Dict]:
    """
    Identify statistical outliers using multiple methods
    """
    if not data or value_field not in data[0]:
        return []

    values = np.array([float(row[value_field]) for row in data if value_field in row])
    if len(values) < 4:
        return []

    outliers = []

    # Method 1: Z-score
    z_scores = np.abs(stats.zscore(values))
    z_outliers = np.where(z_scores > 3)[0]

    # Method 2: IQR
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    iqr_outliers = np.where((values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR))[0]

    # Combine methods
    outlier_indices = set(z_outliers) | set(iqr_outliers)

    for idx in outlier_indices:
        outliers.append({
            "index": int(idx),
            "value": float(values[idx]),
            "z_score": float(z_scores[idx]),
            "data": data[idx]
        })

    return sorted(outliers, key=lambda x: abs(x["z_score"]), reverse=True)


def _compute_distance(vec1: Dict, vec2: Dict, metric: str = "L1") -> float:
    """Helper function to compute distance between two vectors"""
    keys = set(vec1.keys()) | set(vec2.keys())
    v1 = np.array([float(vec1.get(k, 0)) for k in keys])
    v2 = np.array([float(vec2.get(k, 0)) for k in keys])

    if metric == "L2":
        return np.linalg.norm(v1 - v2)
    elif metric in ["COS", "COSINE"]:
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1 - np.dot(v1, v2) / (norm1 * norm2)
    else:  # L1
        return np.sum(np.abs(v1 - v2))