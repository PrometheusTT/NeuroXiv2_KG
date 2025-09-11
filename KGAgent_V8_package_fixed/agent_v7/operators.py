import json
from typing import Dict, Any, List, Tuple
import difflib
from .schema_cache import SchemaCache


def suggest_prop(schema: SchemaCache, label: str, prop: str) -> str:
    """Return best matching existing property for a label when prop doesn't exist."""
    if not schema.has_label(label):
        return prop
    candidates = list(schema.node_props[label].keys())
    match = difflib.get_close_matches(prop, candidates, n=1, cutoff=0.6)
    return match[0] if match else prop


def validate_and_fix_cypher(schema: SchemaCache, query: str) -> str:
    """
    Weak validator that tries to replace missing properties by closest matches per label.
    Only handles simple patterns like (n:Label) and n.prop references.
    """
    # 1) collect aliases -> labels
    # Fix wrong Car3 queries
    if "Subclass {name: 'Car3'}" in query or 'Subclass {name:"Car3"}' in query:
        query = query.replace("Subclass {name: 'Car3'}", "Subclass")
        query = query.replace('Subclass {name:"Car3"}', "Subclass")

        # Add WHERE clause if missing
        if "WHERE" not in query:
            query = query.replace("MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)",
                                  "MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)\nWHERE s.markers CONTAINS 'Car3'")
        else:
            # Add to existing WHERE
            query = query.replace("WHERE ", "WHERE s.markers CONTAINS 'Car3' AND ")

    # Fix Cell node references
    if "Cell" in query or ":Cell" in query:
        query = query.replace("(c:Cell)", "(r:Region)")
        query = query.replace("Cell", "Region")
    import re
    alias_to_label = {}
    for m in re.finditer(r"\((\w+):([A-Za-z_][A-Za-z0-9_]*)\)", query):
        alias, label = m.group(1), m.group(2)
        alias_to_label[alias] = label

    # 2) find alias.prop occurrences
    def repl(match):
        alias, prop = match.group(1), match.group(2)
        label = alias_to_label.get(alias)
        if not label or schema.has_prop(label, prop):
            return f"{alias}.{prop}"
        fixed = suggest_prop(schema, label, prop)
        return f"{alias}.{fixed}"

    query = re.sub(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", repl, query)
    return query


# Few-shot templates you can expand for your KG
NEUROXIV_TEMPLATES = {
    "REGION_MORPH_TOPK": """
MATCH (r:Region)
WHERE r.axonal_length IS NOT NULL AND r.dendritic_length IS NOT NULL
RETURN r.region_id AS region_id, r.acronym AS acronym,
       r.axonal_length AS axon, r.dendritic_length AS dend,
       CASE WHEN r.dendritic_length>0 THEN r.axonal_length/r.dendritic_length ELSE 999 END AS ratio
ORDER BY ratio DESC
LIMIT 50
""",
    "PROJECT_TO_SUMMARY": """
MATCH (a:Region)-[p:PROJECT_TO]->(b:Region)
RETURN a.region_id AS src_id, a.acronym AS src, b.region_id AS dst_id, b.acronym AS dst,
       avg(coalesce(p.weight, p.strength, 0)) AS avg_weight,
       sum(coalesce(p.count, 1)) AS n_edges
ORDER BY avg_weight DESC
LIMIT 100
""",
    "REGION_SUBCLASS_DIST": """
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
RETURN r.region_id AS region_id, r.acronym AS region, s.name AS subclass, coalesce(h.pct_cells, h.weight, h.count, 0) AS value
LIMIT 2000
""",
    "REGION_MORPH_PROPS": """
MATCH (r:Region)
RETURN r.region_id AS region_id, r.acronym AS acronym,
       r.axonal_length AS axonal_length,
       r.dendritic_length AS dendritic_length,
       r.axonal_branches AS axonal_branches,
       r.dendritic_branches AS dendritic_branches,
       r.dendritic_maximum_branch_order AS dendritic_maximum_branch_order
LIMIT 500
""",

    "MORPHOLOGY_MOLECULAR_COMPREHENSIVE": """
MATCH (r:Region)
WHERE r.axonal_length IS NOT NULL AND r.dendritic_length IS NOT NULL
WITH r, 
     r.axonal_length/r.dendritic_length as axon_dendrite_ratio,
     r.axonal_branches as axon_branch,
     r.dendritic_branches as dend_branch
MATCH (r)-[h:HAS_SUBCLASS]->(s:Subclass)
WITH r, axon_dendrite_ratio, axon_branch, dend_branch,
     collect({
         subclass: s.name,
         percentage: h.pct_cells
     }) as molecular_composition
RETURN r.acronym as region,
       r.region_id as region_id,
       axon_dendrite_ratio,
       axon_branch,
       dend_branch,
       molecular_composition
ORDER BY axon_dendrite_ratio DESC
LIMIT 200
""",
    "CONNECTIVITY_HUB_ANALYSIS": """
// Identify connectivity hubs in the brain network
MATCH (r:Region)
WITH r, 
     size((r)-[:PROJECT_TO]->()) as out_degree,
     size((r)<-[:PROJECT_TO]-()) as in_degree
WHERE out_degree > 5 OR in_degree > 5
WITH r, out_degree, in_degree, 
     out_degree + in_degree as total_degree,
     CASE 
         WHEN in_degree > 0 THEN toFloat(out_degree) / in_degree 
         ELSE 999.0 
     END as io_ratio
MATCH (r)-[p:PROJECT_TO]->(target:Region)
WITH r, out_degree, in_degree, total_degree, io_ratio,
     avg(p.weight) as avg_projection_weight,
     collect(DISTINCT target.acronym)[..10] as top_targets
ORDER BY total_degree DESC
RETURN r.acronym as hub_region,
       r.full_name as region_name,
       out_degree,
       in_degree,
       total_degree,
       round(io_ratio, 2) as input_output_ratio,
       round(avg_projection_weight, 3) as avg_weight,
       top_targets
LIMIT 30
""",

    "HIERARCHICAL_ORGANIZATION": """
// Detect hierarchical organization based on projection patterns
MATCH path = (source:Region)-[:PROJECT_TO*1..3]->(target:Region)
WHERE source.acronym IN ['VISp', 'AUDp', 'SSp', 'MOp'] // Primary areas
  AND NOT source = target
WITH source, target, 
     length(path) as path_length,
     [r IN nodes(path) | r.acronym] as path_regions
WITH source.acronym as primary_area,
     target.acronym as target_area,
     min(path_length) as shortest_path,
     collect(DISTINCT path_regions) as all_paths
RETURN primary_area,
       target_area,
       shortest_path as hierarchical_distance,
       size(all_paths) as num_paths,
       all_paths[0] as example_path
ORDER BY primary_area, shortest_path, target_area
LIMIT 100
""",

    "REGION_FINGERPRINT": """
// Generate comprehensive fingerprint for a region
MATCH (r:Region {acronym: $region_acronym})
// Morphological features
WITH r, {
    axonal_length: r.axonal_length,
    dendritic_length: r.dendritic_length,
    axonal_branches: r.axonal_branches,
    dendritic_branches: r.dendritic_branches,
    axonal_bifurcation_angle: r.axonal_bifurcation_remote_angle,
    dendritic_bifurcation_angle: r.dendritic_bifurcation_remote_angle
} as morphology
// Molecular composition
MATCH (r)-[h:HAS_SUBCLASS]->(s:Subclass)
WITH r, morphology, 
     collect({name: s.name, pct: h.pct_cells}) as subclasses
// Projection targets
MATCH (r)-[p:PROJECT_TO]->(target:Region)
WITH r, morphology, subclasses,
     collect({
         target: target.acronym,
         weight: p.weight
     }) as projections
// Projection sources
MATCH (source:Region)-[p2:PROJECT_TO]->(r)
RETURN r.acronym as region,
       r.full_name as full_name,
       morphology,
       subclasses[..10] as top_subclasses,
       projections[..10] as top_projections,
       collect(DISTINCT source.acronym)[..10] as top_inputs
""",
   "CAR3_MARKER_ANALYSIS": """
-- Find regions containing Car3+ neurons (Car3 in marker genes)
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.markers CONTAINS 'Car3' OR 'Car3' IN s.markers
WITH r, s, h.pct_cells as car3_pct
MATCH (r)-[p:PROJECT_TO]->(target:Region)
RETURN 
    r.acronym as source_region,
    s.name as subclass_name,
    s.markers as markers,
    car3_pct as percentage_car3_positive,
    target.acronym as target_region,
    p.weight as projection_weight
ORDER BY car3_pct DESC, projection_weight DESC
LIMIT 100
""",

    "MARKER_GENE_DISTRIBUTION": """
-- Analyze distribution of any marker gene across regions
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.markers CONTAINS $marker_gene  -- e.g., 'Car3', 'Sst', 'Pvalb'
RETURN 
    r.acronym as region,
    s.name as subclass_name,
    s.markers as all_markers,
    h.pct_cells as percentage,
    h.count as cell_count
ORDER BY h.pct_cells DESC
LIMIT 100
""",

    "MULTI_MARKER_ANALYSIS": """
-- Find subclasses with multiple specific markers
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.markers CONTAINS 'Car3' AND s.markers CONTAINS 'Sst'
RETURN 
    r.acronym as region,
    s.name as subclass,
    s.markers as markers,
    h.pct_cells as percentage
ORDER BY h.pct_cells DESC
LIMIT 50
""",

"CAR3_PROJECTION_PATTERN": """
-- Projection patterns of regions with high Car3+ populations
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.markers CONTAINS 'Car3'
WITH r, sum(h.pct_cells) as total_car3_pct
WHERE total_car3_pct > 5  -- Regions with >5% Car3+ cells
MATCH (r)-[p:PROJECT_TO]->(target:Region)
WITH r, total_car3_pct, target, p
ORDER BY total_car3_pct DESC, p.weight DESC
RETURN 
    r.acronym as car3_rich_region,
    total_car3_pct,
    collect(DISTINCT {
        target: target.acronym,
        weight: p.weight
    })[..10] as top_projections
LIMIT 30
"""
}


def generate_hypothesis_test_query(hypothesis: str, entities: List[str],
                                   relationship: str = "PROJECT_TO") -> str:
    """
    Generate a Cypher query to test a specific hypothesis
    """
    if "projection" in hypothesis.lower() and "PROJECT_TO" in relationship:
        return f"""
        MATCH (source:Region)-[p:PROJECT_TO]->(target:Region)
        WHERE source.acronym IN {entities}
        WITH source, target, p.weight as weight
        ORDER BY weight DESC
        RETURN source.acronym as source,
               target.acronym as target,
               weight,
               CASE 
                   WHEN weight > 0.5 THEN 'strong'
                   WHEN weight > 0.2 THEN 'moderate'
                   ELSE 'weak'
               END as connection_strength
        LIMIT 100
        """

    elif "subclass" in hypothesis.lower() and "HAS_SUBCLASS" in relationship:
        return f"""
        MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
        WHERE r.acronym IN {entities} OR s.name IN {entities}
        RETURN r.acronym as region,
               s.name as subclass,
               h.pct_cells as percentage,
               CASE
                   WHEN h.pct_cells > 20 THEN 'dominant'
                   WHEN h.pct_cells > 10 THEN 'major'
                   WHEN h.pct_cells > 5 THEN 'significant'
                   ELSE 'minor'
               END as prevalence
        ORDER BY h.pct_cells DESC
        LIMIT 100
        """

    else:
        # Generic exploration query
        return f"""
        MATCH (n:Region)
        WHERE n.acronym IN {entities}
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.acronym as entity,
               type(r) as relationship,
               labels(m) as target_type,
               CASE
                   WHEN m:Region THEN m.acronym
                   WHEN m:Subclass THEN m.name
                   ELSE id(m)
               END as target,
               r
        LIMIT 200
        """

def example_planner_prompt(schema: SchemaCache) -> str:
    """Corrected prompt with proper marker gene understanding"""
    return f"""
You are analyzing the NeuroXiv knowledge graph. 

## CRITICAL SCHEMA UNDERSTANDING:

### Node Types:
- **Region**: Brain regions (e.g., CLA, RSPagl, MOp)
  - Properties: acronym, full_name, region_id, morphology properties
- **Subclass**: Cell type subclasses
  - Properties: name (subclass name), **markers** (marker genes list)
- **NO Cell nodes exist**

### Relationships:
- `(Region)-[HAS_SUBCLASS]->(Subclass)`: Region contains this cell subclass
  - Properties: pct_cells (percentage), count, weight
- `(Region)-[PROJECT_TO]->(Region)`: Projections between regions
  - Properties: weight, strength

### MARKER GENES (CRITICAL):
- Car3, Sst, Pvalb, etc. are NOT subclass names
- They are MARKER GENES found in the Subclass.markers property
- To find Car3+ neurons, search for Subclasses WHERE markers CONTAINS 'Car3'

## CORRECT QUERY PATTERNS:

✅ CORRECT - Finding Car3+ neurons:
```cypher
MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE 'Car3' IN s.markers OR s.markers CONTAINS 'Car3'
RETURN r.acronym, s.name, h.pct_cells
✅ ALTERNATIVE - If markers is a string:
cypherMATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
WHERE s.markers CONTAINS 'Car3' OR s.markers =~ '.*Car3.*'
RETURN r.acronym, s.name, h.pct_cells
❌ WRONG:
cypherMATCH (s:Subclass {{name: 'Car3'}})  // Car3 is not a subclass name!
Current Schema:
{schema.summary_text()}
When analyzing marker genes like Car3:

Search in the markers property of Subclass nodes
Car3+ neurons means Subclasses that have Car3 in their markers
The analysis should find which Regions contain these Car3+ subclasses
"""
