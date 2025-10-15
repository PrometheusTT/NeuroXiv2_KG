#!/usr/bin/env python3
"""
Test script for the enhanced agent with morphological similarity analysis.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the current directory to the Python path to enable relative imports
sys.path.insert(0, os.path.dirname(__file__))

from agent_v7.enhanced_neo4j_exec import EnhancedNeo4jExec
from agent_v7.morphology_tools import RegionComparisonTools, MorphologicalAnalysisTools, MolecularProfileTools
from agent_v7.schema_cache import SchemaCache

def test_enhanced_agent():
    """Test the enhanced components directly."""
    load_dotenv()

    # Initialize the enhanced components directly
    enhanced_db = EnhancedNeo4jExec(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        pwd=os.getenv('NEO4J_PASSWORD', 'password'),
        database='neo4j'
    )

    # Initialize schema
    schema = SchemaCache()
    with enhanced_db.driver.session() as s:
        schema.load_from_db(s)

    # Initialize specialized tools
    region_comparison = RegionComparisonTools(enhanced_db, schema)
    morph_tools = MorphologicalAnalysisTools(enhanced_db, schema)
    mol_tools = MolecularProfileTools(enhanced_db, schema)

    try:
        # Test the original question
        question = "Search for the most significant pairs of regions that have similar morphological characteristics but significantly different molecular characteristics. And try to explain this"

        print("🧠 Testing Enhanced Morphological-Molecular Analysis")
        print("=" * 80)
        print(f"Question: {question}")
        print()

        # Run the main analysis function
        print("📊 Running comprehensive analysis...")
        result = region_comparison.find_morphologically_similar_molecularly_different_regions(
            morphological_threshold=0.1,
            molecular_threshold=0.3,
            limit=10
        )

        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return

        print("✅ Analysis completed successfully!")
        print()

        methodology = result.get('methodology', {})
        results = result.get('results', {})
        summary = result.get('summary', {})

        print("📋 Methodology:")
        print(f"  Morphological similarity threshold: {methodology.get('morphological_similarity_threshold', 'N/A')}")
        print(f"  Molecular difference threshold: {methodology.get('molecular_difference_threshold', 'N/A')}")
        print(f"  Features analyzed: {methodology.get('morphological_features', [])}")
        print()

        print("📊 Results Summary:")
        print(f"  Morphologically similar pairs found: {results.get('total_morphologically_similar_pairs', 0)}")
        print(f"  Pairs with different neurotransmitters: {results.get('pairs_with_different_neurotransmitters', 0)}")
        print(f"  Final significant pairs: {results.get('final_significant_pairs', 0)}")
        print()

        pairs = results.get('pairs', [])
        if pairs:
            print("🎯 Top Significant Region Pairs:")
            for i, pair in enumerate(pairs[:5], 1):
                print(f"{i}. **{pair['region1']} vs {pair['region2']}**")
                print(f"   Morphological similarity: {pair.get('morphological_similarity', 0):.4f} (lower = more similar)")
                print(f"   Neurotransmitters: {pair.get('region1_dominant_nt', 'N/A')} vs {pair.get('region2_dominant_nt', 'N/A')}")

                if 'molecular_analysis' in pair:
                    mol = pair['molecular_analysis']
                    print(f"   Molecular difference (Jaccard): {mol.get('jaccard_distance', 0):.3f}")
                    print(f"   Shared markers: {len(mol.get('shared_markers', []))}")
                    print(f"   Unique markers total: {mol.get('total_unique_markers', 0)}")
                print()
        else:
            print("No significant pairs found with current thresholds.")
        print()

        # Test individual tools
        print("🔧 Testing Individual Tools:")
        print("-" * 40)

        # Test morphological similarity finder
        print("1. Testing morphological similarity finder...")
        morph_result = morph_tools.find_morphologically_similar_regions(
            similarity_threshold=0.05,
            limit=10
        )

        if 'similar_pairs' in morph_result:
            print(f"   Found {len(morph_result['similar_pairs'])} morphologically similar pairs")
            if morph_result['similar_pairs']:
                top_pair = morph_result['similar_pairs'][0]
                print(f"   Top pair: {top_pair['region1']} vs {top_pair['region2']} (similarity: {top_pair.get('morphological_similarity', 'N/A'):.4f})")
        else:
            print(f"   Error: {morph_result}")
        print()

        # Test neurotransmitter profiling
        print("2. Testing neurotransmitter profiling...")
        regions_to_test = ["ACAv", "ORBvl", "IC", "LGv"]
        nt_result = mol_tools.get_neurotransmitter_profiles(regions_to_test)

        for region in regions_to_test:
            if region in nt_result:
                profile = nt_result[region]
                if profile:
                    top_nt = profile[0]
                    print(f"   {region}: {top_nt['neurotransmitter_type']} ({top_nt['percentage']:.2f}%)")
                else:
                    print(f"   {region}: No neurotransmitter data")
            else:
                print(f"   {region}: Error retrieving data")
        print()

        # Test molecular marker comparison
        print("3. Testing molecular marker comparison...")
        marker_result = mol_tools.compare_molecular_markers("ACAv", "ORBvl", 3)

        if 'jaccard_distance' in marker_result:
            print(f"   ACAv vs ORBvl:")
            print(f"   Jaccard distance: {marker_result['jaccard_distance']:.3f}")
            print(f"   Shared markers: {len(marker_result.get('shared_markers', []))}")
            print(f"   Unique to ACAv: {len(marker_result.get('unique_to_region1', []))}")
            print(f"   Unique to ORBvl: {len(marker_result.get('unique_to_region2', []))}")
        else:
            print(f"   Error: {marker_result}")
        print()

        print("✅ Enhanced agent testing completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        enhanced_db.close()


if __name__ == "__main__":
    test_enhanced_agent()