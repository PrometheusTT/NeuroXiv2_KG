"""
候选回路发现系统 v2.1 - 修复版
适配场景: CLA 没有 Subregion/ME_Subregion 细分

新策略:
1. 分析所有有 Car3 标记的区域（不限于CLA）
2. 寻找有细分结构的其他脑区
3. 通过投射关系分析 CLA 的功能连接
"""

from neo4j import GraphDatabase
import pandas as pd
import numpy as np

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "neuroxiv")

driver = GraphDatabase.driver(URI, auth=AUTH)

# ==== Step 1: 发现所有有Car3标记的分子富集区域 ====

QUERY_FIND_ALL_CAR3_POCKETS = """
// 找到所有空间节点（不限于CLA）及其 Subclass 组成
MATCH (spatial)
WHERE spatial:Region OR spatial:Subregion OR spatial:ME_Subregion

// 获取每个空间节点的 Subclass profile
MATCH (spatial)-[has_sc:HAS_SUBCLASS]->(sc:Subclass)

WITH spatial, 
     labels(spatial) AS spatial_type,
     COALESCE(spatial.acronym, spatial.name) AS spatial_name,
     sc,
     has_sc.pct_cells AS pct_cells,
     has_sc.rank AS rank

// 按空间节点分组
WITH spatial, spatial_type, spatial_name,
     collect({
         subclass_name: sc.name,
         markers: sc.markers,
         pct_cells: pct_cells,
         rank: rank
     }) AS subclass_profile

// 找出 rank=1 的 subclass（主导类型）
WITH spatial, spatial_type, spatial_name, subclass_profile,
     [x IN subclass_profile WHERE x.rank = 1][0] AS dominant_subclass,
     size(subclass_profile) AS n_subclasses

WHERE dominant_subclass IS NOT NULL
  AND (
      // Car3 在标记基因中
      dominant_subclass.markers CONTAINS 'Car3' 
      // 或在 subclass 名称中
      OR dominant_subclass.subclass_name CONTAINS 'Car3'
  )

RETURN 
    elementId(spatial) AS pocket_eid,
    spatial_type,
    spatial_name,
    dominant_subclass.subclass_name AS dominant_subclass_name,
    dominant_subclass.markers AS dominant_markers,
    dominant_subclass.pct_cells AS dominant_pct,
    dominant_subclass.rank AS dominant_rank,
    n_subclasses,
    subclass_profile
ORDER BY 
    // 优先 ME_Subregion
    CASE 
        WHEN 'ME_Subregion' IN spatial_type THEN 1
        WHEN 'Subregion' IN spatial_type THEN 2
        WHEN 'Region' IN spatial_type THEN 3
    END,
    dominant_pct DESC
LIMIT 50
"""

def find_car3_pockets(tx):
    """发现所有 Car3 富集的空间区域"""
    rows = []
    for rec in tx.run(QUERY_FIND_ALL_CAR3_POCKETS):
        rows.append({
            'pocket_eid': rec['pocket_eid'],
            'spatial_type': rec['spatial_type'],
            'spatial_name': rec['spatial_name'],
            'dominant_subclass_name': rec['dominant_subclass_name'],
            'dominant_markers': rec['dominant_markers'],
            'dominant_pct': float(rec['dominant_pct'] or 0.0),
            'n_subclasses': rec['n_subclasses'],
            'granularity_score': 1 if 'ME_Subregion' in str(rec['spatial_type']) else
                                2 if 'Subregion' in str(rec['spatial_type']) else 3
        })
    return pd.DataFrame(rows)

# ==== Step 2: 提取神经元（修复版）====

QUERY_GET_NEURONS_IN_POCKET = """
MATCH (spatial)
WHERE elementId(spatial) = $POCKET_EID

// 根据spatial的类型，使用不同的定位关系
OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT_SUBREGION]->(spatial)
WHERE spatial:Subregion

OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(spatial)
WHERE spatial:ME_Subregion

WITH COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) AS neurons
UNWIND neurons AS n
WITH DISTINCT n
WHERE n IS NOT NULL

RETURN 
    n.neuron_id AS neuron_id,
    n.name AS neuron_name,
    n.celltype AS celltype,
    n.base_region AS base_region
"""

def get_neurons_in_pocket(tx, pocket_eid):
    """提取空间区域中的神经元"""
    rows = []
    for rec in tx.run(QUERY_GET_NEURONS_IN_POCKET, POCKET_EID=pocket_eid):
        rows.append({
            'neuron_id': rec['neuron_id'],
            'neuron_name': rec['neuron_name'],
            'celltype': rec['celltype'],
            'base_region': rec['base_region'],
        })
    return pd.DataFrame(rows)

# ==== Step 3: 分析投射模式（修复版）====

QUERY_GET_PROJECTION_TARGETS = """
MATCH (spatial)
WHERE elementId(spatial) = $POCKET_EID

// 获取神经元
OPTIONAL MATCH (n1:Neuron)-[:LOCATE_AT_SUBREGION]->(spatial)
WHERE spatial:Subregion

OPTIONAL MATCH (n2:Neuron)-[:LOCATE_AT_ME_SUBREGION]->(spatial)
WHERE spatial:ME_Subregion

WITH COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) AS neurons
UNWIND neurons AS n
WITH DISTINCT n
WHERE n IS NOT NULL

// 获取投射
MATCH (n)-[proj:PROJECT_TO]->(target)
WHERE target:Region OR target:Subregion OR target:ME_Subregion

// 尝试获取投射强度（多个可能的属性）
WITH target,
     labels(target) AS target_type,
     COALESCE(target.acronym, target.name) AS target_name,
     n.neuron_id AS source_neuron,
     COALESCE(proj.projection_length, proj.weight, proj.total, 0) AS proj_strength

WHERE proj_strength > 0

// 按目标汇总
WITH target, target_type, target_name,
     count(DISTINCT source_neuron) AS n_contributing_neurons,
     sum(proj_strength) AS total_projection_strength,
     avg(proj_strength) AS avg_projection_strength,
     collect(DISTINCT source_neuron)[0..5] AS sample_neurons

WITH target, target_type, target_name,
     n_contributing_neurons,
     total_projection_strength,
     avg_projection_strength,
     sample_neurons,
     CASE 
         WHEN 'ME_Subregion' IN target_type THEN 1
         WHEN 'Subregion' IN target_type THEN 2
         WHEN 'Region' IN target_type THEN 3
         ELSE 4
     END AS target_granularity

RETURN 
    elementId(target) AS target_eid,
    target_type,
    target_name,
    target_granularity,
    n_contributing_neurons,
    total_projection_strength,
    avg_projection_strength,
    sample_neurons

ORDER BY n_contributing_neurons DESC, total_projection_strength DESC
LIMIT 30
"""

def get_projection_targets(tx, pocket_eid):
    """分析投射目标"""
    rows = []
    for rec in tx.run(QUERY_GET_PROJECTION_TARGETS, POCKET_EID=pocket_eid):
        rows.append({
            'target_eid': rec['target_eid'],
            'target_type': rec['target_type'],
            'target_name': rec['target_name'],
            'target_granularity': rec['target_granularity'],
            'n_contributing_neurons': rec['n_contributing_neurons'],
            'total_projection_strength': rec['total_projection_strength'],
            'avg_projection_strength': rec['avg_projection_strength'],
            'sample_neurons': rec['sample_neurons']
        })
    return pd.DataFrame(rows)

# ==== Step 4: 获取目标分子组成 ====

QUERY_GET_TARGET_MOLECULAR_PROFILE = """
MATCH (target)
WHERE elementId(target) = $TARGET_EID

MATCH (target)-[has_sc:HAS_SUBCLASS]->(sc:Subclass)

RETURN 
    COALESCE(target.acronym, target.name) AS target_name,
    sc.name AS subclass_name,
    sc.markers AS markers,
    has_sc.pct_cells AS pct_cells,
    has_sc.rank AS rank

ORDER BY has_sc.rank ASC
"""

def get_target_molecular_profile(tx, target_eid):
    """获取投射目标的分子组成"""
    rows = []
    for rec in tx.run(QUERY_GET_TARGET_MOLECULAR_PROFILE, TARGET_EID=target_eid):
        rows.append({
            'target_name': rec['target_name'],
            'subclass_name': rec['subclass_name'],
            'markers': rec['markers'],
            'pct_cells': rec['pct_cells'],
            'rank': rec['rank']
        })
    return pd.DataFrame(rows)

# ==== 评分和排名 ====

def rank_candidate_circuits(pockets_df, neurons_dict, projections_dict):
    """候选回路排名"""
    scored_pockets = []

    for idx, row in pockets_df.iterrows():
        pocket_eid = row['pocket_eid']

        score = 0
        reasons = []

        # 1. 分子特异性 (0-30分)
        molecular_score = min(row['dominant_pct'], 30)
        score += molecular_score
        reasons.append(f"分子特异性: {molecular_score:.1f}分 (主导类占比{row['dominant_pct']:.1f}%)")

        # 2. 空间精确性 (0-20分)
        granularity_score = {1: 20, 2: 15, 3: 10}.get(row['granularity_score'], 0)
        score += granularity_score
        spatial_level = {1: 'ME_Subregion', 2: 'Subregion', 3: 'Region'}[row['granularity_score']]
        reasons.append(f"空间精确性: {granularity_score}分 ({spatial_level}级别)")

        # 3. 神经元数量 (0-20分)
        n_neurons = len(neurons_dict.get(pocket_eid, []))
        if 5 <= n_neurons <= 1000:
            neuron_score = 20
        elif n_neurons > 1000:
            neuron_score = 15
        elif n_neurons > 0:
            neuron_score = 10
        else:
            neuron_score = 0
        score += neuron_score
        reasons.append(f"神经元数量: {neuron_score}分 ({n_neurons}个神经元)")

        # 4. 投射特异性 (0-20分)
        projections = projections_dict.get(pocket_eid, pd.DataFrame())
        if not projections.empty:
            top_3_fraction = projections.head(3)['n_contributing_neurons'].sum() / projections['n_contributing_neurons'].sum()
            projection_score = min(top_3_fraction * 20, 20)
            reasons.append(f"投射特异性: {projection_score:.1f}分 (Top3占比{top_3_fraction*100:.1f}%)")
        else:
            projection_score = 0
            reasons.append(f"投射特异性: 0分 (无投射数据)")
        score += projection_score

        # 5. Car3 相关性 (0-10分) - 已经是Car3区域了，都得满分
        car3_score = 10
        reasons.append(f"Car3相关性: 10分 (Car3-like)")
        score += car3_score

        scored_pockets.append({
            'pocket_eid': pocket_eid,
            'spatial_name': row['spatial_name'],
            'spatial_type': row['spatial_type'],
            'total_score': score,
            'molecular_score': molecular_score,
            'granularity_score': granularity_score,
            'neuron_score': neuron_score,
            'projection_score': projection_score,
            'car3_score': car3_score,
            'n_neurons': n_neurons,
            'n_projection_targets': len(projections),
            'ranking_reasons': ' | '.join(reasons)
        })

    return pd.DataFrame(scored_pockets).sort_values('total_score', ascending=False)

# ==== 主流程 ====

def main():
    print("="*70)
    print("候选回路发现系统 v2.1 (Car3 全脑扫描版)")
    print("策略: 扫描全脑所有 Car3 富集区域")
    print("="*70)

    with driver.session() as session:

        # Step 1: 发现所有 Car3 富集区域
        print("\n[Step 1] 扫描全脑 Car3 富集区域...")
        pockets_df = session.execute_read(find_car3_pockets)
        print(f"  发现 {len(pockets_df)} 个 Car3 富集区域")

        if not pockets_df.empty:
            print(f"  - ME_Subregion级别: {len(pockets_df[pockets_df['granularity_score']==1])} 个")
            print(f"  - Subregion级别: {len(pockets_df[pockets_df['granularity_score']==2])} 个")
            print(f"  - Region级别: {len(pockets_df[pockets_df['granularity_score']==3])} 个")

            pockets_df.to_csv("step1_car3_pockets_全脑.csv", index=False)
            print(f"\n[✓] 保存: step1_car3_pockets_全脑.csv")
        else:
            print("[ERROR] 未发现任何 Car3 富集区域！")
            return

        # Step 2-3: 提取神经元和投射
        print("\n[Step 2-3] 提取神经元和投射模式...")

        neurons_dict = {}
        projections_dict = {}

        for idx, pocket in pockets_df.iterrows():
            pocket_eid = pocket['pocket_eid']
            pocket_name = pocket['spatial_name']

            print(f"\n  处理 {idx+1}/{len(pockets_df)}: {pocket_name}")

            # 提取神经元
            neurons_df = session.execute_read(get_neurons_in_pocket, pocket_eid)
            neurons_dict[pocket_eid] = neurons_df
            print(f"    - 神经元: {len(neurons_df)}")

            # 提取投射
            if not neurons_df.empty:
                projections_df = session.execute_read(get_projection_targets, pocket_eid)
                projections_dict[pocket_eid] = projections_df
                print(f"    - 投射目标: {len(projections_df)}")
            else:
                projections_dict[pocket_eid] = pd.DataFrame()

        # Step 4: 排名
        print("\n[Step 4] 候选回路排名...")
        ranked_circuits = rank_candidate_circuits(pockets_df, neurons_dict, projections_dict)
        ranked_circuits.to_csv("step2_ranked_car3_circuits.csv", index=False)
        print(f"[✓] 保存: step2_ranked_car3_circuits.csv")

        print("\n  Top 10 Car3 候选回路:")
        for idx, circuit in ranked_circuits.head(10).iterrows():
            print(f"\n  [{idx+1}] {circuit['spatial_name']} (总分: {circuit['total_score']:.1f})")
            print(f"      类型: {circuit['spatial_type']}")
            print(f"      神经元: {circuit['n_neurons']} 个")
            print(f"      投射目标: {circuit['n_projection_targets']} 个")

        # Step 5: Top回路详细报告
        if not ranked_circuits.empty:
            print("\n[Step 5] 生成Top回路详细报告...")

            top_circuit = ranked_circuits.iloc[0]
            top_eid = top_circuit['pocket_eid']

            # 保存详细数据
            if top_eid in neurons_dict and not neurons_dict[top_eid].empty:
                neurons_dict[top_eid].to_csv("step3_top_car3_circuit_neurons.csv", index=False)
                print(f"[✓] 保存神经元数据")

            if top_eid in projections_dict and not projections_dict[top_eid].empty:
                projections_dict[top_eid].to_csv("step3_top_car3_circuit_projections.csv", index=False)
                print(f"[✓] 保存投射数据")

                # 获取下游目标的分子profile
                print(f"\n  分析下游目标分子组成...")
                target_profiles = []

                for _, proj in projections_dict[top_eid].head(5).iterrows():
                    target_eid = proj['target_eid']
                    target_name = proj['target_name']
                    print(f"    - {target_name}")

                    target_profile = session.execute_read(get_target_molecular_profile, target_eid)
                    if not target_profile.empty:
                        target_profile['target_eid'] = target_eid
                        target_profiles.append(target_profile)

                if target_profiles:
                    target_profiles_df = pd.concat(target_profiles, ignore_index=True)
                    target_profiles_df.to_csv("step3_top_car3_circuit_target_profiles.csv", index=False)
                    print(f"[✓] 保存下游目标分子组成")

        print("\n" + "="*70)
        print("分析完成！")
        print("="*70)
        print("\n生成的文件:")
        print("  1. step1_car3_pockets_全脑.csv - 所有 Car3 富集区域")
        print("  2. step2_ranked_car3_circuits.csv - 候选回路排名")
        print("  3. step3_top_car3_circuit_neurons.csv - Top回路神经元")
        print("  4. step3_top_car3_circuit_projections.csv - Top回路投射")
        print("  5. step3_top_car3_circuit_target_profiles.csv - 下游分子组成")

        # 打印摘要
        if not ranked_circuits.empty:
            top = ranked_circuits.iloc[0]
            print("\n" + "="*70)
            print("Top Car3 回路摘要:")
            print("="*70)
            print(f"起源: {top['spatial_name']} ({top['spatial_type']})")
            print(f"总分: {top['total_score']:.1f}")
            print(f"神经元: {top['n_neurons']} 个")
            print(f"投射目标: {top['n_projection_targets']} 个")
            print(f"\n评分详情:")
            print(f"  {top['ranking_reasons']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()