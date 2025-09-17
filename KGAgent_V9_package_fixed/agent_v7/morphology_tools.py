"""
Specialized tools for morphological similarity analysis and molecular profile comparisons.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import json

from .enhanced_neo4j_exec import EnhancedNeo4jExec
from .schema_cache import SchemaCache

logger = logging.getLogger(__name__)


class MorphologicalAnalysisTools:
    """Tools for analyzing morphological similarities between brain regions."""

    def __init__(self, db: EnhancedNeo4jExec, schema: SchemaCache):
        self.db = db
        self.schema = schema

    def find_morphologically_similar_regions(self, similarity_threshold: float = 0.1,
                                           limit: int = 50) -> Dict[str, Any]:
        """
        Find pairs of regions with similar morphological characteristics.
        Uses relative differences to handle different scales.
        """
        query = """
        MATCH (r1:Region)
        MATCH (r2:Region)
        WHERE r1.name < r2.name
          AND (r1.axonal_length > 1000 OR r1.dendritic_length > 1000)
          AND (r2.axonal_length > 1000 OR r2.dendritic_length > 1000)

        WITH r1, r2,
             abs(coalesce(r1.axonal_length, 0) - coalesce(r2.axonal_length, 0)) /
             (coalesce(r1.axonal_length, 0) + coalesce(r2.axonal_length, 0) + 1) AS axon_rel_diff,
             abs(coalesce(r1.dendritic_length, 0) - coalesce(r2.dendritic_length, 0)) /
             (coalesce(r1.dendritic_length, 0) + coalesce(r2.dendritic_length, 0) + 1) AS dend_rel_diff,
             abs(coalesce(r1.axonal_branches, 0) - coalesce(r2.axonal_branches, 0)) /
             (coalesce(r1.axonal_branches, 0) + coalesce(r2.axonal_branches, 0) + 1) AS axon_branch_rel_diff,
             abs(coalesce(r1.dendritic_branches, 0) - coalesce(r2.dendritic_branches, 0)) /
             (coalesce(r1.dendritic_branches, 0) + coalesce(r2.dendritic_branches, 0) + 1) AS dend_branch_rel_diff

        WITH r1, r2,
             (axon_rel_diff + dend_rel_diff + axon_branch_rel_diff + dend_branch_rel_diff) / 4 AS avg_rel_diff

        WHERE avg_rel_diff <= $similarity_threshold

        RETURN r1.name AS region1, r2.name AS region2,
               avg_rel_diff AS morphological_similarity,
               r1.axonal_length AS r1_axonal, r1.dendritic_length AS r1_dendritic,
               r1.axonal_branches AS r1_axonal_branches, r1.dendritic_branches AS r1_dendritic_branches,
               r2.axonal_length AS r2_axonal, r2.dendritic_length AS r2_dendritic,
               r2.axonal_branches AS r2_axonal_branches, r2.dendritic_branches AS r2_dendritic_branches
        ORDER BY avg_rel_diff ASC
        LIMIT $limit
        """

        result = self.db.run_direct(query, {
            "similarity_threshold": similarity_threshold,
            "limit": limit
        })

        if not result["success"]:
            return {"error": "Morphological similarity query failed", "details": result}

        return {
            "similar_pairs": result["data"],
            "total_pairs": len(result["data"]),
            "similarity_threshold": similarity_threshold,
            "morphological_features": [
                "axonal_length", "dendritic_length",
                "axonal_branches", "dendritic_branches"
            ]
        }

    def compute_z_score_morphological_distance(self, region_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Compute z-score normalized morphological distances between region pairs.
        This is more sophisticated than relative differences.
        """
        # First get population statistics
        stats_query = """
        MATCH (r:Region)
        WHERE r.axonal_length IS NOT NULL OR r.dendritic_length IS NOT NULL
        WITH collect({
            axonal_length: coalesce(r.axonal_length, 0.0),
            dendritic_length: coalesce(r.dendritic_length, 0.0),
            axonal_branches: coalesce(r.axonal_branches, 0.0),
            dendritic_branches: coalesce(r.dendritic_branches, 0.0),
            axonal_maximum_branch_order: coalesce(r.axonal_maximum_branch_order, 0.0),
            dendritic_maximum_branch_order: coalesce(r.dendritic_maximum_branch_order, 0.0)
        }) AS all_regions

        UNWIND all_regions AS r
        RETURN
            avg(r.axonal_length) AS mean_axonal_length,
            stDev(r.axonal_length) AS std_axonal_length,
            avg(r.dendritic_length) AS mean_dendritic_length,
            stDev(r.dendritic_length) AS std_dendritic_length,
            avg(r.axonal_branches) AS mean_axonal_branches,
            stDev(r.axonal_branches) AS std_axonal_branches,
            avg(r.dendritic_branches) AS mean_dendritic_branches,
            stDev(r.dendritic_branches) AS std_dendritic_branches,
            avg(r.axonal_maximum_branch_order) AS mean_axonal_max_order,
            stDev(r.axonal_maximum_branch_order) AS std_axonal_max_order,
            avg(r.dendritic_maximum_branch_order) AS mean_dendritic_max_order,
            stDev(r.dendritic_maximum_branch_order) AS std_dendritic_max_order
        """

        stats_result = self.db.run_direct(stats_query)
        if not stats_result["success"] or not stats_result["data"]:
            return {"error": "Failed to compute population statistics"}

        stats = stats_result["data"][0]

        # Now compute distances for each pair
        pair_distances = []
        for region1, region2 in region_pairs:
            pair_query = """
            MATCH (r1:Region {name: $region1})
            MATCH (r2:Region {name: $region2})
            RETURN
                r1.name AS region1,
                r1.axonal_length AS r1_axonal_length,
                r1.dendritic_length AS r1_dendritic_length,
                r1.axonal_branches AS r1_axonal_branches,
                r1.dendritic_branches AS r1_dendritic_branches,
                r1.axonal_maximum_branch_order AS r1_axonal_max_order,
                r1.dendritic_maximum_branch_order AS r1_dendritic_max_order,
                r2.name AS region2,
                r2.axonal_length AS r2_axonal_length,
                r2.dendritic_length AS r2_dendritic_length,
                r2.axonal_branches AS r2_axonal_branches,
                r2.dendritic_branches AS r2_dendritic_branches,
                r2.axonal_maximum_branch_order AS r2_axonal_max_order,
                r2.dendritic_maximum_branch_order AS r2_dendritic_max_order
            """

            pair_result = self.db.run_direct(pair_query, {
                "region1": region1,
                "region2": region2
            })

            if pair_result["success"] and pair_result["data"]:
                row = pair_result["data"][0]

                # Compute z-scores and distance
                features = [
                    'axonal_length', 'dendritic_length', 'axonal_branches',
                    'dendritic_branches', 'axonal_max_order', 'dendritic_max_order'
                ]

                z_distance = 0.0
                feature_details = {}

                for feature in features:
                    r1_val = row[f'r1_{feature}'] or 0.0
                    r2_val = row[f'r2_{feature}'] or 0.0
                    mean_val = stats[f'mean_{feature}'] or 0.0
                    std_val = stats[f'std_{feature}'] or 1.0

                    if std_val > 0:
                        z1 = (r1_val - mean_val) / std_val
                        z2 = (r2_val - mean_val) / std_val
                        z_diff = abs(z1 - z2)
                        z_distance += z_diff ** 2
                        feature_details[feature] = {
                            'region1_value': r1_val,
                            'region2_value': r2_val,
                            'region1_zscore': z1,
                            'region2_zscore': z2,
                            'zscore_diff': z_diff
                        }

                z_distance = np.sqrt(z_distance)

                pair_distances.append({
                    'region1': region1,
                    'region2': region2,
                    'z_score_distance': z_distance,
                    'feature_details': feature_details
                })

        return {
            'pair_distances': pair_distances,
            'population_stats': stats,
            'methodology': 'z_score_euclidean_distance'
        }


class MolecularProfileTools:
    """Tools for analyzing molecular profiles and neurotransmitter characteristics."""

    def __init__(self, db: EnhancedNeo4jExec, schema: SchemaCache):
        self.db = db
        self.schema = schema

    def get_neurotransmitter_profiles(self, region_names: List[str]) -> Dict[str, Any]:
        """Get detailed neurotransmitter profiles for specified regions."""
        profiles = {}

        for region_name in region_names:
            query = """
            MATCH (r:Region {name: $region_name})-[h:HAS_SUBCLASS]->(s:Subclass)
            WHERE s.dominant_neurotransmitter_type IS NOT NULL
            WITH r, s.dominant_neurotransmitter_type AS nt,
                 sum(coalesce(h.pct_cells, 0.0)) AS total_pct
            ORDER BY total_pct DESC
            RETURN nt AS neurotransmitter_type, total_pct AS percentage
            """

            result = self.db.run_direct(query, {"region_name": region_name})

            if result["success"] and result["data"]:
                profiles[region_name] = result["data"]
            else:
                profiles[region_name] = []

        return profiles

    def find_different_neurotransmitter_pairs(self, morphologically_similar_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find pairs from morphologically similar regions that have different dominant neurotransmitters.
        """
        different_nt_pairs = []

        for pair in morphologically_similar_pairs:
            region1 = pair['region1']
            region2 = pair['region2']

            # Get dominant neurotransmitter for each region
            query = """
            MATCH (r1:Region {name: $region1})-[h1:HAS_SUBCLASS]->(s1:Subclass)
            MATCH (r2:Region {name: $region2})-[h2:HAS_SUBCLASS]->(s2:Subclass)
            WHERE s1.dominant_neurotransmitter_type IS NOT NULL
              AND s2.dominant_neurotransmitter_type IS NOT NULL

            WITH r1, r2, s1.dominant_neurotransmitter_type AS nt1, h1.pct_cells AS pct1,
                 s2.dominant_neurotransmitter_type AS nt2, h2.pct_cells AS pct2

            WITH r1, r2, nt1, sum(pct1) AS total_pct1, nt2, sum(pct2) AS total_pct2
            ORDER BY r1.name, r2.name, total_pct1 DESC, total_pct2 DESC

            WITH r1, r2,
                 collect(DISTINCT {nt: nt1, pct: total_pct1})[0] AS dom_nt1,
                 collect(DISTINCT {nt: nt2, pct: total_pct2})[0] AS dom_nt2

            WHERE dom_nt1.nt <> dom_nt2.nt

            RETURN r1.name AS region1, r2.name AS region2,
                   dom_nt1.nt AS region1_dominant_nt,
                   dom_nt2.nt AS region2_dominant_nt,
                   dom_nt1.pct AS region1_nt_percentage,
                   dom_nt2.pct AS region2_nt_percentage
            """

            result = self.db.run_direct(query, {
                "region1": region1,
                "region2": region2
            })

            if result["success"] and result["data"]:
                nt_info = result["data"][0]
                different_nt_pairs.append({
                    **pair,  # Include original morphological similarity data
                    **nt_info  # Add neurotransmitter information
                })

        return {
            "different_neurotransmitter_pairs": different_nt_pairs,
            "total_pairs": len(different_nt_pairs)
        }

    def get_molecular_markers_profile(self, region_name: str, top_n: int = 5) -> Dict[str, Any]:
        """Get detailed molecular marker profiles for a region."""
        query = """
        MATCH (r:Region {name: $region_name})-[h:HAS_SUBCLASS]->(s:Subclass)
        RETURN
            r.name AS region,
            s.name AS subclass,
            s.dominant_neurotransmitter_type AS neurotransmitter,
            h.pct_cells AS pct_cells,
            s.markers[0..10] AS top_markers,
            s.transcription_factor_markers[0..5] AS tf_markers
        ORDER BY h.pct_cells DESC
        LIMIT $top_n
        """

        result = self.db.run_direct(query, {
            "region_name": region_name,
            "top_n": top_n
        })

        if result["success"]:
            return {
                "region": region_name,
                "top_subclasses": result["data"],
                "total_subclasses_analyzed": len(result["data"])
            }
        else:
            return {"error": f"Failed to get molecular markers for {region_name}"}

    def compare_molecular_markers(self, region1: str, region2: str, top_n: int = 3) -> Dict[str, Any]:
        """Compare molecular marker profiles between two regions."""
        # Get markers for both regions
        profile1 = self.get_molecular_markers_profile(region1, top_n)
        profile2 = self.get_molecular_markers_profile(region2, top_n)

        if "error" in profile1 or "error" in profile2:
            return {"error": "Failed to retrieve marker profiles"}

        # Extract all markers from top subclasses
        markers1 = set()
        markers2 = set()

        for subclass in profile1["top_subclasses"]:
            if subclass["top_markers"]:
                markers1.update(subclass["top_markers"])
            if subclass["tf_markers"]:
                markers1.update(subclass["tf_markers"])

        for subclass in profile2["top_subclasses"]:
            if subclass["top_markers"]:
                markers2.update(subclass["top_markers"])
            if subclass["tf_markers"]:
                markers2.update(subclass["tf_markers"])

        # Compute Jaccard similarity
        intersection = markers1.intersection(markers2)
        union = markers1.union(markers2)

        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        jaccard_distance = 1.0 - jaccard_similarity

        return {
            "region1": region1,
            "region2": region2,
            "region1_markers": list(markers1),
            "region2_markers": list(markers2),
            "shared_markers": list(intersection),
            "unique_to_region1": list(markers1 - markers2),
            "unique_to_region2": list(markers2 - markers1),
            "jaccard_similarity": jaccard_similarity,
            "jaccard_distance": jaccard_distance,
            "marker_overlap_percentage": len(intersection) / max(len(markers1), len(markers2)) * 100 if max(len(markers1), len(markers2)) > 0 else 0,
            "total_unique_markers": len(union)
        }


class RegionComparisonTools:
    """Integrated tools for comprehensive region comparisons."""

    def __init__(self, db: EnhancedNeo4jExec, schema: SchemaCache):
        self.db = db
        self.schema = schema
        self.morph_tools = MorphologicalAnalysisTools(db, schema)
        self.mol_tools = MolecularProfileTools(db, schema)

    def find_morphologically_similar_molecularly_different_regions(
            self,
            morphological_threshold: float = 0.1,
            molecular_threshold: float = 0.3,
            limit: int = 20
    ) -> Dict[str, Any]:
        """
        Main analysis function: find regions with similar morphology but different molecular profiles.
        """
        # Step 1: Find morphologically similar regions
        logger.info("Finding morphologically similar regions...")
        morph_similar = self.morph_tools.find_morphologically_similar_regions(
            similarity_threshold=morphological_threshold,
            limit=limit * 5  # Get more candidates
        )

        if "error" in morph_similar:
            return morph_similar

        # Step 2: Find those with different neurotransmitter profiles
        logger.info("Filtering for different neurotransmitter profiles...")
        nt_different = self.mol_tools.find_different_neurotransmitter_pairs(
            morph_similar["similar_pairs"]
        )

        if not nt_different["different_neurotransmitter_pairs"]:
            return {"message": "No pairs found with different neurotransmitter profiles"}

        # Step 3: Analyze molecular marker differences
        logger.info("Analyzing molecular marker differences...")
        final_pairs = []
        for pair in nt_different["different_neurotransmitter_pairs"][:limit]:
            marker_comparison = self.mol_tools.compare_molecular_markers(
                pair["region1"], pair["region2"]
            )

            if "error" not in marker_comparison:
                # Only include pairs with significant molecular differences
                if marker_comparison["jaccard_distance"] >= molecular_threshold:
                    final_pairs.append({
                        **pair,
                        "molecular_analysis": marker_comparison
                    })

        return {
            "methodology": {
                "morphological_similarity_threshold": morphological_threshold,
                "molecular_difference_threshold": molecular_threshold,
                "morphological_features": morph_similar.get("morphological_features", []),
                "analysis_steps": [
                    "Compute morphological similarity using relative differences",
                    "Filter for different dominant neurotransmitter types",
                    "Analyze molecular marker overlap using Jaccard distance"
                ]
            },
            "results": {
                "total_morphologically_similar_pairs": len(morph_similar["similar_pairs"]),
                "pairs_with_different_neurotransmitters": len(nt_different["different_neurotransmitter_pairs"]),
                "final_significant_pairs": len(final_pairs),
                "pairs": final_pairs
            },
            "summary": {
                "found_significant_pairs": len(final_pairs) > 0,
                "most_similar_morphologically": final_pairs[0] if final_pairs else None,
                "average_morphological_similarity": np.mean([p["morphological_similarity"] for p in final_pairs]) if final_pairs else None,
                "average_molecular_difference": np.mean([p["molecular_analysis"]["jaccard_distance"] for p in final_pairs]) if final_pairs else None
            }
        }

    def detailed_region_comparison(self, region1: str, region2: str) -> Dict[str, Any]:
        """Provide detailed comparison between two specific regions."""
        # Morphological comparison
        z_score_analysis = self.morph_tools.compute_z_score_morphological_distance(
            [(region1, region2)]
        )

        # Neurotransmitter profiles
        nt_profiles = self.mol_tools.get_neurotransmitter_profiles([region1, region2])

        # Molecular marker comparison
        marker_analysis = self.mol_tools.compare_molecular_markers(region1, region2)

        return {
            "region_pair": [region1, region2],
            "morphological_analysis": z_score_analysis,
            "neurotransmitter_profiles": nt_profiles,
            "molecular_marker_analysis": marker_analysis,
            "interpretation": self._generate_interpretation(
                z_score_analysis, nt_profiles, marker_analysis, region1, region2
            )
        }

    def _generate_interpretation(self, morph_analysis: Dict, nt_profiles: Dict,
                               marker_analysis: Dict, region1: str, region2: str) -> Dict[str, Any]:
        """Generate biological interpretation of the comparison."""
        interpretation = {
            "morphological_similarity": "unknown",
            "molecular_difference": "unknown",
            "biological_significance": "unknown"
        }

        # Morphological interpretation
        if morph_analysis.get("pair_distances"):
            z_distance = morph_analysis["pair_distances"][0]["z_score_distance"]
            if z_distance < 1.0:
                interpretation["morphological_similarity"] = "Very similar - likely similar developmental constraints or functional requirements"
            elif z_distance < 2.0:
                interpretation["morphological_similarity"] = "Moderately similar - some shared architectural features"
            else:
                interpretation["morphological_similarity"] = "Quite different morphologically"

        # Molecular interpretation
        if marker_analysis.get("jaccard_distance"):
            jaccard_dist = marker_analysis["jaccard_distance"]
            if jaccard_dist > 0.7:
                interpretation["molecular_difference"] = "Highly distinct molecular profiles - likely different developmental origins and functions"
            elif jaccard_dist > 0.5:
                interpretation["molecular_difference"] = "Moderately different molecular signatures"
            else:
                interpretation["molecular_difference"] = "Similar molecular characteristics"

        # Combined biological significance
        if (interpretation["morphological_similarity"].startswith("Very similar") and
            interpretation["molecular_difference"].startswith("Highly distinct")):
            interpretation["biological_significance"] = (
                "Fascinating example of morphological convergence with molecular divergence - "
                "suggests similar computational requirements solved by different cell types/developmental programs"
            )

        return interpretation