"""
Benchmark Question Set
=======================
分层测试问题集

难度分级：
- Level 1: 简单查询（单实体、单模态）
- Level 2: 中等分析（多实体、跨模态）
- Level 3: 复杂推理（闭环分析、多模态整合）

问题类型：
- marker_query: 基因标记物查询
- region_analysis: 脑区分析
- comparison: 对比分析
- projection_tracing: 投射追踪
- circuit_analysis: 环路分析

Author: Lijun
Date: 2025-01
"""

from typing import Dict, List, Any

# ==================== Level 1: Simple Queries ====================

LEVEL_1_QUESTIONS = [
    {
        "id": "L1_01",
        "question": "What clusters express Vip as a marker gene?",
        "level": 1,
        "type": "marker_query",
        "expected_intent": "marker_query",
        "expected_entities": {"genes": ["Vip"]},
        "expected_modalities": ["molecular"],
        "expected_keywords": ["cluster", "Vip", "neuron"],
        "ground_truth": {
            "key_facts": ["Vip is expressed in GABAergic interneurons",
                          "Multiple clusters including Vip_Cluster_1, Vip_Cluster_2"],
            "expected_clusters": ["Vip"]
        }
    },
    {
        "id": "L1_02",
        "question": "What is the full name of brain region VISp?",
        "level": 1,
        "type": "region_analysis",
        "expected_intent": "region_info",
        "expected_entities": {"regions": ["VISp"]},
        "expected_modalities": ["spatial"],
        "expected_keywords": ["primary", "visual", "cortex", "VISp"],
        "ground_truth": {
            "key_facts": ["VISp = Primary Visual Cortex"],
            "full_name": "Primary Visual Area"
        }
    },
    {
        "id": "L1_03",
        "question": "Which clusters have Sst as a marker?",
        "level": 1,
        "type": "marker_query",
        "expected_intent": "marker_query",
        "expected_entities": {"genes": ["Sst"]},
        "expected_modalities": ["molecular"],
        "expected_keywords": ["Sst", "cluster", "somatostatin"],
        "ground_truth": {
            "key_facts": ["Sst marks inhibitory interneurons"],
            "expected_clusters": ["Sst"]
        }
    },
    {
        "id": "L1_04",
        "question": "What neurons are located in MOp?",
        "level": 1,
        "type": "region_analysis",
        "expected_intent": "region_composition",
        "expected_entities": {"regions": ["MOp"]},
        "expected_modalities": ["molecular", "spatial"],
        "expected_keywords": ["MOp", "motor", "neuron", "cluster"],
        "ground_truth": {
            "key_facts": ["MOp = Primary Motor Area", "Contains excitatory and inhibitory neurons"]
        }
    },
    {
        "id": "L1_05",
        "question": "Tell me about Pvalb-expressing cells",
        "level": 1,
        "type": "marker_query",
        "expected_intent": "marker_query",
        "expected_entities": {"genes": ["Pvalb"]},
        "expected_modalities": ["molecular"],
        "expected_keywords": ["Pvalb", "parvalbumin", "interneuron", "fast-spiking"],
        "ground_truth": {
            "key_facts": ["Pvalb marks fast-spiking interneurons", "GABAergic"]
        }
    },
    {
        "id": "L1_06",
        "question": "What is ACA in the brain?",
        "level": 1,
        "type": "region_analysis",
        "expected_intent": "region_info",
        "expected_entities": {"regions": ["ACA"]},
        "expected_modalities": ["spatial"],
        "expected_keywords": ["anterior", "cingulate", "ACA"],
        "ground_truth": {
            "key_facts": ["ACA = Anterior Cingulate Area"]
        }
    },
    {
        "id": "L1_07",
        "question": "Which genes mark L5 IT neurons?",
        "level": 1,
        "type": "marker_query",
        "expected_intent": "cluster_markers",
        "expected_entities": {"cell_types": ["L5 IT"]},
        "expected_modalities": ["molecular"],
        "expected_keywords": ["L5", "IT", "marker", "gene"],
        "ground_truth": {
            "key_facts": ["L5 IT = Layer 5 Intratelencephalic neurons"]
        }
    },
    {
        "id": "L1_08",
        "question": "How many neurons express Car3?",
        "level": 1,
        "type": "marker_query",
        "expected_intent": "marker_query",
        "expected_entities": {"genes": ["Car3"]},
        "expected_modalities": ["molecular"],
        "expected_keywords": ["Car3", "neuron", "number", "count"],
        "ground_truth": {
            "key_facts": ["Car3 is a marker gene"]
        }
    },
]

# ==================== Level 2: Medium Analysis ====================

LEVEL_2_QUESTIONS = [
    {
        "id": "L2_01",
        "question": "Compare Vip+ and Sst+ neurons in terms of their distribution and markers",
        "level": 2,
        "type": "comparison",
        "expected_intent": "comparison",
        "expected_entities": {"genes": ["Vip", "Sst"]},
        "expected_modalities": ["molecular", "spatial"],
        "expected_keywords": ["Vip", "Sst", "distribution", "marker", "difference", "interneuron"],
        "ground_truth": {
            "key_facts": [
                "Both are GABAergic interneuron markers",
                "Different layer distributions",
                "Distinct co-expression patterns"
            ]
        }
    },
    {
        "id": "L2_02",
        "question": "What are the projection targets of neurons in VISp?",
        "level": 2,
        "type": "projection_tracing",
        "expected_intent": "projection_analysis",
        "expected_entities": {"regions": ["VISp"]},
        "expected_modalities": ["projection", "spatial"],
        "expected_keywords": ["VISp", "projection", "target", "connect"],
        "ground_truth": {
            "key_facts": [
                "VISp projects to higher visual areas",
                "Targets include VISl, VISpm, VISam"
            ]
        }
    },
    {
        "id": "L2_03",
        "question": "What is the morphological characteristics of neurons in ACA?",
        "level": 2,
        "type": "region_analysis",
        "expected_intent": "morphology_analysis",
        "expected_entities": {"regions": ["ACA"]},
        "expected_modalities": ["morphological", "spatial"],
        "expected_keywords": ["ACA", "morphology", "axon", "dendrite", "structure"],
        "ground_truth": {
            "key_facts": [
                "Contains pyramidal neurons with long-range axons",
                "Layer-specific morphological features"
            ]
        }
    },
    {
        "id": "L2_04",
        "question": "Which clusters in MOp have the highest neuron counts and what are their markers?",
        "level": 2,
        "type": "region_analysis",
        "expected_intent": "region_composition",
        "expected_entities": {"regions": ["MOp"]},
        "expected_modalities": ["molecular", "spatial"],
        "expected_keywords": ["MOp", "cluster", "neuron", "count", "marker"],
        "ground_truth": {
            "key_facts": [
                "MOp contains multiple cell types",
                "Excitatory and inhibitory populations"
            ]
        }
    },
    {
        "id": "L2_05",
        "question": "How do L2/3 IT and L5 IT neurons differ in their gene expression?",
        "level": 2,
        "type": "comparison",
        "expected_intent": "comparison",
        "expected_entities": {"cell_types": ["L2/3 IT", "L5 IT"]},
        "expected_modalities": ["molecular"],
        "expected_keywords": ["L2/3", "L5", "IT", "gene", "expression", "difference"],
        "ground_truth": {
            "key_facts": [
                "Layer-specific gene expression patterns",
                "Different projection targets"
            ]
        }
    },
    {
        "id": "L2_06",
        "question": "What brain regions receive projections from MOp and what cell types are in those regions?",
        "level": 2,
        "type": "projection_tracing",
        "expected_intent": "projection_analysis",
        "expected_entities": {"regions": ["MOp"]},
        "expected_modalities": ["projection", "molecular"],
        "expected_keywords": ["MOp", "projection", "target", "cell", "type"],
        "ground_truth": {
            "key_facts": [
                "MOp projects to multiple motor-related areas",
                "Targets include striatum, thalamus"
            ]
        }
    },
    {
        "id": "L2_07",
        "question": "Describe the Lamp5+ neurons including their location and co-expressed markers",
        "level": 2,
        "type": "marker_query",
        "expected_intent": "marker_query",
        "expected_entities": {"genes": ["Lamp5"]},
        "expected_modalities": ["molecular", "spatial"],
        "expected_keywords": ["Lamp5", "location", "marker", "co-express", "neuron"],
        "ground_truth": {
            "key_facts": [
                "Lamp5 marks specific interneuron subtypes",
                "Found in superficial layers"
            ]
        }
    },
    {
        "id": "L2_08",
        "question": "What is the relationship between Ndnf expression and neuronal morphology?",
        "level": 2,
        "type": "marker_query",
        "expected_intent": "cross_modal_analysis",
        "expected_entities": {"genes": ["Ndnf"]},
        "expected_modalities": ["molecular", "morphological"],
        "expected_keywords": ["Ndnf", "morphology", "axon", "dendrite", "expression"],
        "ground_truth": {
            "key_facts": [
                "Ndnf marks neurogliaform cells",
                "Distinct morphological features"
            ]
        }
    },
]

# ==================== Level 3: Complex Analysis ====================

LEVEL_3_QUESTIONS = [
    {
        "id": "L3_01",
        "question": "Analyze the complete circuit from VISp: what are the projection targets, what cell types are in those targets, and what are their molecular markers?",
        "level": 3,
        "type": "circuit_analysis",
        "expected_intent": "circuit_analysis",
        "expected_entities": {"regions": ["VISp"]},
        "expected_modalities": ["molecular", "projection", "spatial"],
        "expected_keywords": ["VISp", "circuit", "projection", "target", "cell", "marker", "pathway"],
        "ground_truth": {
            "key_facts": [
                "VISp is source of visual processing circuit",
                "Projects to higher visual areas",
                "Target regions contain specific cell types",
                "Complete pathway analysis"
            ]
        }
    },
    {
        "id": "L3_02",
        "question": "Compare the projection patterns and morphological features of Vip+ vs Sst+ interneurons across cortical regions",
        "level": 3,
        "type": "comparison",
        "expected_intent": "cross_modal_comparison",
        "expected_entities": {"genes": ["Vip", "Sst"]},
        "expected_modalities": ["molecular", "morphological", "projection"],
        "expected_keywords": ["Vip", "Sst", "projection", "morphology", "cortex", "compare"],
        "ground_truth": {
            "key_facts": [
                "Vip and Sst have different morphologies",
                "Different projection patterns",
                "Layer-specific distributions"
            ]
        }
    },
    {
        "id": "L3_03",
        "question": "Trace the motor circuit: starting from MOp, identify projection targets, analyze the cell composition of targets, and describe the morphological features of the projecting neurons",
        "level": 3,
        "type": "circuit_analysis",
        "expected_intent": "circuit_analysis",
        "expected_entities": {"regions": ["MOp"]},
        "expected_modalities": ["molecular", "morphological", "projection"],
        "expected_keywords": ["MOp", "motor", "circuit", "projection", "target", "morphology", "cell"],
        "ground_truth": {
            "key_facts": [
                "MOp is primary motor cortex",
                "Projects to subcortical motor structures",
                "Contains projection neurons with distinct morphology"
            ]
        }
    },
    {
        "id": "L3_04",
        "question": "What are the molecular, morphological, and connectivity differences between L5 ET and L5 IT neurons?",
        "level": 3,
        "type": "comparison",
        "expected_intent": "cross_modal_comparison",
        "expected_entities": {"cell_types": ["L5 ET", "L5 IT"]},
        "expected_modalities": ["molecular", "morphological", "projection"],
        "expected_keywords": ["L5", "ET", "IT", "molecular", "morphology", "connectivity", "projection"],
        "ground_truth": {
            "key_facts": [
                "L5 ET projects extratelencephalically",
                "L5 IT projects intratelencephalically",
                "Different morphological and molecular profiles"
            ]
        }
    },
    {
        "id": "L3_05",
        "question": "Analyze the ACA circuit: projection targets, target compositions, and identify potential marker genes that could distinguish ACA-projecting vs non-projecting neurons",
        "level": 3,
        "type": "circuit_analysis",
        "expected_intent": "circuit_analysis",
        "expected_entities": {"regions": ["ACA"]},
        "expected_modalities": ["molecular", "projection", "spatial"],
        "expected_keywords": ["ACA", "circuit", "projection", "marker", "distinguish", "target"],
        "ground_truth": {
            "key_facts": [
                "ACA connects limbic and motor systems",
                "Multiple projection targets",
                "Distinct molecular signatures"
            ]
        }
    },
    {
        "id": "L3_06",
        "question": "Perform a comprehensive analysis of Car3+ neurons: their subclass distribution, regional preferences, morphological features, and projection patterns",
        "level": 3,
        "type": "marker_query",
        "expected_intent": "comprehensive_analysis",
        "expected_entities": {"genes": ["Car3"]},
        "expected_modalities": ["molecular", "morphological", "projection", "spatial"],
        "expected_keywords": ["Car3", "subclass", "region", "morphology", "projection", "comprehensive"],
        "ground_truth": {
            "key_facts": [
                "Car3 marks specific neuronal populations",
                "Multi-modal characterization",
                "Distribution and connectivity patterns"
            ]
        }
    },
    {
        "id": "L3_07",
        "question": "Map the visual-motor pathway: from VISp through intermediate stations to MOp, describing cell types and connectivity at each stage",
        "level": 3,
        "type": "circuit_analysis",
        "expected_intent": "pathway_analysis",
        "expected_entities": {"regions": ["VISp", "MOp"]},
        "expected_modalities": ["molecular", "projection", "spatial"],
        "expected_keywords": ["visual", "motor", "pathway", "VISp", "MOp", "connectivity", "stage"],
        "ground_truth": {
            "key_facts": [
                "Visual-motor integration pathway",
                "Multi-synaptic connections",
                "Intermediate processing stations"
            ]
        }
    },
    {
        "id": "L3_08",
        "question": "Compare three interneuron types (Vip+, Sst+, Pvalb+) across all modalities: molecular signatures, morphological features, regional distribution, and connectivity patterns",
        "level": 3,
        "type": "comparison",
        "expected_intent": "multi_comparison",
        "expected_entities": {"genes": ["Vip", "Sst", "Pvalb"]},
        "expected_modalities": ["molecular", "morphological", "projection", "spatial"],
        "expected_keywords": ["Vip", "Sst", "Pvalb", "interneuron", "compare", "morphology", "connectivity"],
        "ground_truth": {
            "key_facts": [
                "Three major interneuron classes",
                "Distinct molecular profiles",
                "Different morphological and connectivity features"
            ]
        }
    },
]

# ==================== Combined Question Set ====================

ALL_QUESTIONS = LEVEL_1_QUESTIONS + LEVEL_2_QUESTIONS + LEVEL_3_QUESTIONS


def get_questions_by_level(level: int) -> List[Dict]:
    """获取指定难度的问题"""
    if level == 1:
        return LEVEL_1_QUESTIONS
    elif level == 2:
        return LEVEL_2_QUESTIONS
    elif level == 3:
        return LEVEL_3_QUESTIONS
    else:
        return ALL_QUESTIONS


def get_questions_by_type(qtype: str) -> List[Dict]:
    """获取指定类型的问题"""
    return [q for q in ALL_QUESTIONS if q.get('type') == qtype]


def get_balanced_sample(n_per_level: int = 3) -> List[Dict]:
    """获取平衡采样"""
    import random
    sample = []
    for level in [1, 2, 3]:
        questions = get_questions_by_level(level)
        sample.extend(random.sample(questions, min(n_per_level, len(questions))))
    return sample


# ==================== Statistics ====================

def get_question_stats() -> Dict:
    """获取问题集统计"""
    stats = {
        'total': len(ALL_QUESTIONS),
        'by_level': {
            1: len(LEVEL_1_QUESTIONS),
            2: len(LEVEL_2_QUESTIONS),
            3: len(LEVEL_3_QUESTIONS),
        },
        'by_type': {},
    }

    for q in ALL_QUESTIONS:
        qtype = q.get('type', 'unknown')
        stats['by_type'][qtype] = stats['by_type'].get(qtype, 0) + 1

    return stats


# ==================== Export ====================

__all__ = [
    'LEVEL_1_QUESTIONS',
    'LEVEL_2_QUESTIONS',
    'LEVEL_3_QUESTIONS',
    'ALL_QUESTIONS',
    'get_questions_by_level',
    'get_questions_by_type',
    'get_balanced_sample',
    'get_question_stats',
]