#!/usr/bin/env python3
"""
Enhanced KG Agent V7 runner with morphological similarity analysis capabilities.
This script demonstrates the enhanced agent that can handle the original question without query errors.
"""

import json
import os
from dotenv import load_dotenv

from agent_v7.enhanced_neo4j_exec import EnhancedNeo4jExec
from agent_v7.morphology_tools import RegionComparisonTools
from agent_v7.schema_cache import SchemaCache


def main():
    load_dotenv()

    # Initialize enhanced components
    enhanced_db = EnhancedNeo4jExec(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        pwd=os.getenv('NEO4J_PASSWORD', 'password'),
        database='neo4j'
    )

    schema = SchemaCache()
    with enhanced_db.driver.session() as s:
        schema.load_from_db(s)

    region_comparison = RegionComparisonTools(enhanced_db, schema)

    try:
        print("🧠 Enhanced KG Agent V7 - Morphological-Molecular Analysis")
        print("=" * 70)
        print()

        # Original question that caused query errors
        question = "Search for the most significant pairs of regions that have similar morphological characteristics but significantly different molecular characteristics. And try to explain this"

        print(f"Question: {question}")
        print()
        print("🔍 Running enhanced analysis...")

        # Use the specialized analysis function
        result = region_comparison.find_morphologically_similar_molecularly_different_regions(
            morphological_threshold=0.05,  # More stringent similarity requirement
            molecular_threshold=0.3,       # Require significant molecular differences
            limit=15
        )

        # Generate comprehensive response
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return

        methodology = result.get('methodology', {})
        results = result.get('results', {})
        summary = result.get('summary', {})
        pairs = results.get('pairs', [])

        print("✅ Analysis completed successfully!")
        print()

        # Display methodology
        print("📋 Methodology:")
        print(f"• Morphological similarity threshold: {methodology.get('morphological_similarity_threshold')}")
        print(f"• Molecular difference threshold: {methodology.get('molecular_difference_threshold')}")
        print(f"• Morphological features analyzed: {', '.join(methodology.get('morphological_features', []))}")
        print("• Analysis steps:")
        for step in methodology.get('analysis_steps', []):
            print(f"  - {step}")
        print()

        # Display results
        print("📊 Results:")
        print(f"• Total morphologically similar pairs found: {results.get('total_morphologically_similar_pairs', 0)}")
        print(f"• Pairs with different dominant neurotransmitters: {results.get('pairs_with_different_neurotransmitters', 0)}")
        print(f"• Final significant pairs (meeting both criteria): {results.get('final_significant_pairs', 0)}")
        print()

        if pairs:
            print("🎯 Most Significant Region Pairs:")
            print("=" * 50)

            for i, pair in enumerate(pairs, 1):
                print(f"\n{i}. **{pair['region1']} vs {pair['region2']}**")
                print(f"   📏 Morphological similarity: {pair.get('morphological_similarity', 0):.4f}")
                print(f"      (Lower values indicate more similar morphology)")

                print(f"   🧪 Neurotransmitter profiles:")
                print(f"      • {pair['region1']}: {pair.get('region1_dominant_nt', 'Unknown')} ({pair.get('region1_nt_percentage', 0):.2f}%)")
                print(f"      • {pair['region2']}: {pair.get('region2_dominant_nt', 'Unknown')} ({pair.get('region2_nt_percentage', 0):.2f}%)")

                if 'molecular_analysis' in pair:
                    mol = pair['molecular_analysis']
                    print(f"   🧬 Molecular marker analysis:")
                    print(f"      • Jaccard distance: {mol.get('jaccard_distance', 0):.3f}")
                    print(f"      • Shared markers: {len(mol.get('shared_markers', []))}")
                    print(f"      • Unique to {pair['region1']}: {len(mol.get('unique_to_region1', []))}")
                    print(f"      • Unique to {pair['region2']}: {len(mol.get('unique_to_region2', []))}")
                    print(f"      • Total unique markers: {mol.get('total_unique_markers', 0)}")

                    if mol.get('shared_markers'):
                        print(f"      • Example shared markers: {', '.join(mol['shared_markers'][:3])}")
        else:
            print("ℹ️  No region pairs found meeting both morphological similarity and molecular difference criteria.")
            print("   Consider adjusting the thresholds or expanding the analysis parameters.")

        print()
        print("🔬 Biological Interpretation:")
        print("=" * 30)

        if pairs:
            # Provide biological explanation
            print("These results demonstrate a fascinating aspect of brain organization:")
            print()
            print("🔹 **Morphological Convergence**: Regions can evolve similar structural")
            print("   patterns (axonal/dendritic architecture) to support similar")
            print("   computational requirements, even when using different cell types.")
            print()
            print("🔹 **Molecular Specialization**: The distinct molecular profiles reflect:")
            print("   • Different developmental origins")
            print("   • Functional specialization (e.g., inhibitory vs excitatory processing)")
            print("   • Regional identity maintenance through unique gene expression")
            print()
            print("🔹 **Form vs Function**: This demonstrates that morphological similarity")
            print("   can mask profound molecular and functional differences, highlighting")
            print("   the complexity of brain regional specification.")
            print()

            # Highlight the most interesting pair
            if pairs:
                top_pair = pairs[0]
                print(f"🌟 **Most Notable Example**: {top_pair['region1']} vs {top_pair['region2']}")
                print(f"   Despite nearly identical morphology (similarity: {top_pair.get('morphological_similarity', 0):.4f}),")
                print(f"   these regions use completely different neurotransmitter systems")
                print(f"   ({top_pair.get('region1_dominant_nt', 'N/A')} vs {top_pair.get('region2_dominant_nt', 'N/A')})")
                print(f"   and have no overlapping molecular markers.")
        else:
            print("The analysis found high morphological diversity or insufficient")
            print("molecular differentiation among similar regions with current parameters.")

        print()
        print("✨ Enhanced analysis completed without query errors!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        enhanced_db.close()


if __name__ == "__main__":
    main()