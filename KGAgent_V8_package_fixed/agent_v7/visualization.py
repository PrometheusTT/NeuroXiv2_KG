"""
Visualization utilities for AIPOM-CoT results
Generates plots similar to paper figures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import networkx as nx


def visualize_projection_pattern(projection_data: Dict) -> Dict:
    """
    Create visualization data for projection patterns (like Figure 3)
    """
    viz_data = {
        "nodes": [],
        "edges": [],
        "clusters": [],
        "layout": "hierarchical"
    }

    # Extract nodes and edges
    for item in projection_data.get("hub_regions", []):
        viz_data["nodes"].append({
            "id": item["region"],
            "label": item["region"],
            "size": item["score"],
            "type": "hub",
            "degree": item["out_degree"] + item["in_degree"]
        })

        for target, weight in item.get("top_targets", []):
            viz_data["edges"].append({
                "source": item["region"],
                "target": target,
                "weight": weight,
                "type": "projection"
            })

    return viz_data


def visualize_mismatch_matrix(mismatch_data: Dict) -> Dict:
    """
    Create heatmap data for morphology-molecular mismatch (like Figure 4B)
    """
    matrix = mismatch_data.get("mismatch_matrix", [])
    regions = mismatch_data.get("regions", [])

    viz_data = {
        "matrix": matrix,
        "labels": regions,
        "colormap": "RdBu_r",
        "title": "Morphology-Molecular Mismatch Index",
        "xlabel": "Brain Regions",
        "ylabel": "Brain Regions"
    }

    return viz_data


def generate_analysis_report_html(results: Dict) -> str:
    """
    Generate HTML report with embedded visualizations
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AIPOM-CoT Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
            .metric {{ background: #ecf0f1; padding: 10px; margin: 10px 0; }}
            .hypothesis {{ background: #e8f6f3; padding: 15px; margin: 15px 0; }}
            .finding {{ background: #fef9e7; padding: 12px; margin: 12px 0; }}
            pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>AIPOM-CoT Analysis: {question}</h1>

        <h2>Hypotheses Tested</h2>
        {hypotheses_html}

        <h2>Key Discoveries</h2>
        {discoveries_html}

        <h2>Statistical Findings</h2>
        {statistics_html}

        <h2>Network Analysis</h2>
        {network_html}

        <h2>Final Insights</h2>
        <div class="finding">
            {final_answer}
        </div>
    </body>
    </html>
    """

    # Format sections
    hypotheses_html = "\n".join([
        f'<div class="hypothesis"><strong>{h["claim"]}</strong><br>'
        f'Confidence: {h.get("confidence", "N/A")}</div>'
        for h in results.get("hypotheses", [])
    ])

    discoveries_html = "\n".join([
        f'<div class="finding">{insight}</div>'
        for insight in results.get("discoveries", {}).get("insights", [])
    ])

    # Format final HTML
    return html_template.format(
        question=results.get("question", ""),
        hypotheses_html=hypotheses_html,
        discoveries_html=discoveries_html,
        statistics_html="<pre>" + json.dumps(results.get("discoveries", {}).get("statistical_findings", []),
                                             indent=2) + "</pre>",
        network_html="<pre>" + json.dumps(results.get("discoveries", {}).get("patterns", {}), indent=2) + "</pre>",
        final_answer=results.get("final_answer", "").replace("\n", "<br>")
    )