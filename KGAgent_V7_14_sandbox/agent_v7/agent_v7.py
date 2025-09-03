# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
import pandas as pd

from .planner import Planner
from .neo4j_exec import Graph
from . import metrics as M
from . import viz

class KGAgentV7:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pwd: str, database: str,
                 openai_api_key: str = "", planner_model: str = "gpt-4o", summarizer_model: str = "gpt-4o"):
        self.graph = Graph(neo4j_uri, neo4j_user, neo4j_pwd, database)
        self.planner = Planner(openai_api_key=openai_api_key, planner_model=planner_model, summarizer_model=summarizer_model)

    # --------- Operators over KG ---------
    def fetch_morph_table(self) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (r:Region)
        WHERE (r.axonal_length IS NOT NULL OR r.dendritic_length IS NOT NULL OR
               r.axonal_branches IS NOT NULL OR r.dendritic_branches IS NOT NULL OR
               r.axonal_maximum_branch_order IS NOT NULL OR r.dendritic_maximum_branch_order IS NOT NULL)
        RETURN
          coalesce(r.acronym, r.name) AS region,
          coalesce(r.axonal_length, 0.0) AS axonal_length,
          coalesce(r.dendritic_length, 0.0) AS dendritic_length,
          coalesce(r.axonal_branches, 0.0) AS axonal_branches,
          coalesce(r.dendritic_branches, 0.0) AS dendritic_branches,
          coalesce(r.axonal_maximum_branch_order, 0.0) AS axonal_maximum_branch_order,
          coalesce(r.dendritic_maximum_branch_order, 0.0) AS dendritic_maximum_branch_order
        """
        return self.graph.run(cypher)

    def fetch_region_subclass_table(self) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (r:Region)-[h:HAS_SUBCLASS]->(s:Subclass)
        RETURN
          coalesce(r.acronym, r.name) AS region,
          s.name AS subclass,
          coalesce(h.pct_cells, h.weight, h.pct, h.count, 0.0) AS value
        """
        return self.graph.run(cypher)

    def fetch_projection_edges(self) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (src:Region)-[p]->(dst:Region)
        WHERE any(x IN [type(p)] WHERE
            toLower(x) CONTAINS 'project' OR
            toLower(x) CONTAINS 'connect' OR
            toLower(x) CONTAINS 'target' OR
            toLower(x) CONTAINS 'innervat')
        RETURN
          coalesce(src.acronym, src.name) AS source,
          coalesce(dst.acronym, dst.name) AS target,
          coalesce(p.weight, p.strength, p.count, 1.0) AS strength
        """
        return self.graph.run(cypher)

    # --------- Analyze pipeline ---------
    def analyze(self, question: str, rounds: int = 2, plot: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {"plan": {}, "rounds": 0, "tables": {}, "metrics": {}, "plots": [], "final": "", "cot": {"think": "", "act": [], "observe": [], "reflect": []}}

        # THINK: plan
        plan = self.planner.plan(question)
        result["plan"] = plan
        result["cot"]["think"] = "Plan derived by planner (LLM-first, fallback generic) strictly using KG operators."
        
        tables: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}
        plots: List[str] = []

        for r in range(rounds):
            result["rounds"] = r+1
            # ACT: execute steps
            result["cot"]["act"].append({"round": r+1, "steps": plan.get("steps", [])})
            for step in plan.get("steps", []):
                op = step.get("op")
                args = step.get("args", {}) or {}

                if op == "fetch_morph_table":
                    rows = self.fetch_morph_table()
                    tables["morph_rows"] = rows
                    result["cot"]["observe"].append({"round": r+1, "fetch_morph_table": len(rows)})
                elif op == "fetch_region_subclass_table":
                    rows = self.fetch_region_subclass_table()
                    tables["mole_rows"] = rows
                    result["cot"]["observe"].append({"round": r+1, "fetch_region_subclass_table": len(rows)})
                elif op == "fetch_projection_edges":
                    rows = self.fetch_projection_edges()
                    tables["proj_edges"] = rows
                    result["cot"]["observe"].append({"round": r+1, "fetch_projection_edges": len(rows)})
                elif op == "compute_modality_matrices":
                    morph_df, Xm = M.build_morph_matrix(tables.get("morph_rows", []))
                    mole_df, Xg = M.build_mole_matrix(tables.get("mole_rows", []))
                    regions, morph_df2, mole_df2 = M.align_modalities(morph_df, mole_df)
                    Dm = M.cosine_distance_matrix(M.zscore(morph_df2.to_numpy())) if len(regions)>1 else np.zeros((len(regions), len(regions)))
                    # For molecular, default use cosine on tf-idf already normalized
                    Dg = M.cosine_distance_matrix(mole_df2.to_numpy()) if len(regions)>1 else np.zeros((len(regions), len(regions)))
                    metrics["regions"] = regions
                    metrics["Dm_qc"] = M.qc_stats(Dm)
                    metrics["Dg_qc"] = M.qc_stats(Dg)
                    metrics["Dm"] = Dm.tolist() if Dm.size>0 else []
                    metrics["Dg"] = Dg.tolist() if Dg.size>0 else []
                    result["cot"]["observe"].append({"round": r+1, "compute_modality_matrices": {"regions": len(regions), "Dm_qc": metrics["Dm_qc"], "Dg_qc": metrics["Dg_qc"]}})
                    if plot and len(regions) > 1:
                        p = os.path.join("outputs", "morph_vs_mole_scatter.png")
                        out = viz.scatter_morph_vs_mole(Dm, Dg, p, title="Morph vs Molecular distances")
                        if out: plots.append(out)
                elif op == "mismatch_pairs":
                    regions = metrics.get("regions", [])
                    Dm = np.array(metrics.get("Dm", []), dtype=float) if "Dm" in metrics else None
                    Dg = np.array(metrics.get("Dg", []), dtype=float) if "Dg" in metrics else None
                    pairs = M.list_top_pairs(Dm, Dg, regions, morph_max=0.25, mole_min=0.6, topk=30)
                    metrics["mismatch_pairs"] = pairs
                    result["cot"]["observe"].append({"round": r+1, "mismatch_pairs": len(pairs)})
                    if plot and pairs:
                        # Make a heatmap for top 20 deltas if possible
                        top_regs = sorted(set([p["region_a"] for p in pairs[:20]] + [p["region_b"] for p in pairs[:20]]))
                        idx = [regions.index(r) for r in top_regs if r in regions]
                        if idx and "Dm" in metrics and "Dg" in metrics:
                            Dm = np.array(metrics["Dm"], dtype=float)
                            Dg = np.array(metrics["Dg"], dtype=float)
                            Mi = M.mismatch_index_matrix(Dm, Dg)
                            if Mi is not None:
                                sub = Mi[np.ix_(idx, idx)]
                                p = os.path.join("outputs", "mismatch_heatmap.png")
                                out = viz.heatmap_matrix(sub, top_regs, p, title="Mismatch index (|Dg-Dm|)")
                                if out: plots.append(out)
                elif op == "analyze_subclass_projection_corr":
                    mole_df, _ = M.build_mole_matrix(tables.get("mole_rows", []))
                    proj_strength = M.compute_projection_strength(tables.get("proj_edges", []))
                    corr_df = M.subclass_projection_correlation(mole_df, proj_strength)
                    metrics["subclass_projection_corr"] = corr_df.head(50).to_dict(orient="records")
                    result["cot"]["observe"].append({"round": r+1, "analyze_subclass_projection_corr": len(metrics["subclass_projection_corr"])})
                    if plot and not corr_df.empty:
                        p = os.path.join("outputs", "subclass_projection_top.png")
                        out = viz.bar_top_corr(corr_df, k=min(20, len(corr_df)), out_path=p, title="Subclass-Projection correlation (top)")
                        if out: plots.append(out)
                elif op == "rank_regions_by_subclass":
                    keyword = (args.get("keyword") or "").lower()
                    df = pd.DataFrame(tables.get("mole_rows", []))
                    if not df.empty:
                        for a in ["s.name","subclass_name"]:
                            if a in df.columns and "subclass" not in df.columns:
                                df = df.rename(columns={a:"subclass"})
                        for a in ["r.acronym","r.name"]:
                            if a in df.columns and "region" not in df.columns:
                                df = df.rename(columns={a:"region"})
                        if "value" not in df.columns:
                            for a in ["pct_cells","weight","pct","count"]:
                                if a in df.columns:
                                    df = df.rename(columns={a: "value"}); break
                        if "subclass" in df.columns and "region" in df.columns:
                            sel = df[df["subclass"].str.lower().str.contains(keyword, na=False)]
                            rank = sel.groupby("region", as_index=False)["value"].sum().sort_values("value", ascending=False).head(20)
                            metrics["rank_regions_by_subclass"] = {"keyword": keyword, "top": rank.to_dict(orient="records")}
                            result["cot"]["observe"].append({"round": r+1, "rank_regions_by_subclass": len(rank)})
                        else:
                            metrics["rank_regions_by_subclass"] = {"keyword": keyword, "top": []}
                    else:
                        metrics["rank_regions_by_subclass"] = {"keyword": keyword, "top": []}
                else:
                    # unknown op: skip
                    result["cot"]["observe"].append({"round": r+1, "unknown_op": op})

            # REFLECT: simple reflection based on observations
            reflection = {}
            if "Dm_qc" in metrics or "Dg_qc" in metrics:
                reflection["distance_qc"] = {"Dm_qc": metrics.get("Dm_qc"), "Dg_qc": metrics.get("Dg_qc")}
            if "subclass_projection_corr" in metrics:
                top_corr = metrics["subclass_projection_corr"][:3]
                reflection["projection_corr_hint"] = {"top3": top_corr}
            if "mismatch_pairs" in metrics:
                reflection["mismatch_hint"] = {"n": len(metrics["mismatch_pairs"])}
            result["cot"]["reflect"].append({"round": r+1, "note": reflection})

        # Fill tables sizes for summary
        result["tables"]["morph_rows"] = len(tables.get("morph_rows", [])) if "morph_rows" in tables else 0
        result["tables"]["mole_rows"] = len(tables.get("mole_rows", [])) if "mole_rows" in tables else 0
        if "regions" in metrics:
            result["tables"]["aligned_regions"] = len(metrics["regions"])

        result["metrics"] = metrics
        result["plots"] = plots

        # FINAL narrative strictly based on computed metrics/tables and the question
        result["final"] = self._narrative(question, tables, metrics, plots)

        return result

    def _narrative(self, question: str, tables: Dict[str, Any], metrics: Dict[str, Any], plots: List[str]) -> str:
        lines = []
        lines.append(f"(1) Task: {question}")
        # Data coverage
        lines.append(f"(2) Data: morphology rows={len(tables.get('morph_rows', []))}, region-subclass rows={len(tables.get('mole_rows', []))}")

        # If we computed cross-modality
        if "Dm_qc" in metrics or "Dg_qc" in metrics:
            dmq = metrics.get("Dm_qc", {})
            dgq = metrics.get("Dg_qc", {})
            if dmq:
                lines.append(f"(3) Morph distance QC: min={dmq['min']:.4f}, max={dmq['max']:.4f}, mean={dmq['mean']:.4f}, median={dmq['median']:.4f}, std={dmq['std']:.4f}, n_pairs={dmq['n_pairs']}")
            if dgq:
                lines.append(f"(4) Molecular distance QC: min={dgq['min']:.4f}, max={dgq['max']:.4f}, mean={dgq['mean']:.4f}, median={dgq['median']:.4f}, std={dgq['std']:.4f}, n_pairs={dgq['n_pairs']}")

        # If correlation present and question mentions projection
        qlower = question.lower()
        if "subclass_projection_corr" in metrics and any(k in qlower for k in ["projection", "connect", "target", "innervat", "predict"]):
            top = metrics["subclass_projection_corr"][:5]
            if top:
                lines.append("(5) Subclass—projection correlation (top):")
                for row in top:
                    lines.append(f"    - {row['subclass']}: r={row['pearson_r']:.3f} (n_regions={row['n_regions']})")

        # Mismatch pairs only if user asked for that & we computed it
        if "mismatch_pairs" in metrics and ("mismatch" in qlower or ("similar" in qlower and "morpholog" in qlower and "molecular" in qlower)):
            pairs = metrics["mismatch_pairs"][:5]
            if pairs:
                lines.append("(6) Example region pairs with similar morphology but different molecular profiles:")
                for p in pairs:
                    lines.append(f"    - {p['region_a']} vs {p['region_b']}: D_morph={p['morph_dist']:.3f}, D_mole={p['mole_dist']:.3f}, Δ={p['delta']:.3f}")

        # rank by subclass if present
        if "rank_regions_by_subclass" in metrics and metrics["rank_regions_by_subclass"].get("top"):
            kw = metrics["rank_regions_by_subclass"]["keyword"]
            lines.append(f"(7) Regions enriched for '{kw}' subclasses (top):")
            for r in metrics["rank_regions_by_subclass"]["top"][:5]:
                lines.append(f"    - {r['region']}: {r['value']:.3f}")

        # Plots
        if plots:
            lines.append("(8) Plots saved:")
            for p in plots:
                lines.append(f"    - {p}")

        # Reflection hint
        lines.append("(9) Next-step reflection:")
        if "subclass_projection_corr" in metrics and metrics["subclass_projection_corr"]:
            lines.append("    - Validate top subclass signals by inspecting their spatial distribution across strongly projecting regions; consider adding partial correlations controlling for region size or cell counts (from KG fields if available).")
        elif "Dm_qc" in metrics or "Dg_qc" in metrics:
            lines.append("    - Probe specific region clusters where morphology and molecular distances disagree; test robustness by switching to Jensen–Shannon distance for molecular profiles.")
        else:
            lines.append("    - Enrich the plan with additional KG operators (e.g., cell-type counts or pathway annotations) if present in the graph and re-run.")

        return "\n".join(lines)
