"""
诊断脚本V2：支持扁平化列表格式的CCF树

修复内容：
- 支持 tree_yzx.json 的列表格式（1327个节点）
- 通过 structure_id_path 重建层级关系
"""

import json
import pandas as pd
import numpy as np
import nrrd
from pathlib import Path
from collections import defaultdict
from loguru import logger

logger.add("region_aggregation_diagnosis_v2.log", rotation="10 MB")


def load_ccf_tree_from_list(ccf_tree_json: Path) -> tuple:
    """
    从扁平化列表格式加载CCF树
    
    返回:
        (id_to_node, id_to_acronym, acronym_to_id)
    """
    with open(ccf_tree_json, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list format, got {type(data)}")
    
    logger.info(f"  加载了 {len(data)} 个节点（列表格式）")
    
    # 构建基本映射
    id_to_node = {}
    id_to_acronym = {}
    acronym_to_id = {}
    
    for node in data:
        node_id = node.get('id')
        acronym = node.get('acronym', '')
        
        if node_id and acronym:
            id_to_node[node_id] = node
            id_to_acronym[node_id] = acronym
            acronym_to_id[acronym] = node_id
    
    logger.info(f"  ✓ 构建了 {len(id_to_acronym)} 个ID映射")
    
    return id_to_node, id_to_acronym, acronym_to_id


def build_hierarchy_from_paths(id_to_node: dict) -> tuple:
    """
    从 structure_id_path 重建层级关系
    
    返回:
        (id_to_parent_id, id_to_children_ids, acronym_to_children, acronym_to_all_descendants)
    """
    id_to_parent_id = {}
    id_to_children_ids = defaultdict(list)
    
    logger.info("  通过 structure_id_path 重建层级关系...")
    
    for node_id, node in id_to_node.items():
        path = node.get('structure_id_path', [])
        
        if not path or len(path) == 0:
            logger.warning(f"    节点 {node_id} 没有 structure_id_path")
            continue
        
        # path的最后一个应该是自己
        if path[-1] != node_id:
            logger.warning(f"    节点 {node_id} 的path不一致: {path}")
            continue
        
        # path的倒数第二个是父节点
        if len(path) >= 2:
            parent_id = path[-2]
            id_to_parent_id[node_id] = parent_id
            id_to_children_ids[parent_id].append(node_id)
    
    logger.info(f"  ✓ 重建了 {len(id_to_parent_id)} 个父子关系")
    logger.info(f"  ✓ {len(id_to_children_ids)} 个节点有children")
    
    # 转换为acronym映射
    id_to_acronym = {nid: node.get('acronym') for nid, node in id_to_node.items()}
    
    acronym_to_children = defaultdict(set)
    for parent_id, children_ids in id_to_children_ids.items():
        parent_acronym = id_to_acronym.get(parent_id)
        if parent_acronym:
            for child_id in children_ids:
                child_acronym = id_to_acronym.get(child_id)
                if child_acronym:
                    acronym_to_children[parent_acronym].add(child_acronym)
    
    # 计算所有后代
    acronym_to_all_descendants = defaultdict(set)
    
    def get_all_descendants(acronym: str) -> set:
        if acronym in acronym_to_all_descendants:
            return acronym_to_all_descendants[acronym]
        
        descendants = set()
        for child in acronym_to_children.get(acronym, []):
            descendants.add(child)
            descendants.update(get_all_descendants(child))
        
        acronym_to_all_descendants[acronym] = descendants
        return descendants
    
    # 预计算所有后代
    for acronym in acronym_to_children.keys():
        get_all_descendants(acronym)
    
    logger.info(f"  ✓ 计算了 {len(acronym_to_all_descendants)} 个节点的后代关系")
    
    return id_to_parent_id, id_to_children_ids, acronym_to_children, acronym_to_all_descendants


def map_cells_to_ccf_simple(cells_df: pd.DataFrame, 
                              annotation_volume: np.ndarray,
                              voxel_size: float = 25.0) -> pd.DataFrame:
    """简单的细胞到CCF ID映射（不做任何聚合）"""
    cells = cells_df.copy()
    
    # 计算体素索引
    x_idx = np.round(cells['x_ccf'] / voxel_size).astype(int)
    y_idx = np.round(cells['y_ccf'] / voxel_size).astype(int)
    z_idx = np.round(cells['z_ccf'] / voxel_size).astype(int)
    
    # 限制在边界内
    shape = annotation_volume.shape
    x_idx = np.clip(x_idx, 0, shape[0] - 1)
    y_idx = np.clip(y_idx, 0, shape[1] - 1)
    z_idx = np.clip(z_idx, 0, shape[2] - 1)
    
    # 获取CCF ID
    ccf_ids = annotation_volume[x_idx, y_idx, z_idx]
    cells['ccf_id'] = ccf_ids
    
    return cells


def check_aggregation_behavior(data_path: Path):
    """检查细胞数据是否被聚合到父region"""
    
    logger.info("=" * 80)
    logger.info("开始诊断：Region Children聚合行为 (V2 - 支持列表格式)")
    logger.info("=" * 80)
    
    # ==================== Step 1: 加载CCF树（列表格式）====================
    logger.info("\nStep 1: 加载CCF树（列表格式）...")
    
    ccf_tree_json = data_path / "tree_yzx.json"
    logger.info(f"  文件: {ccf_tree_json}")
    
    id_to_node, id_to_acronym, acronym_to_id = load_ccf_tree_from_list(ccf_tree_json)
    
    # 重建层级关系
    id_to_parent_id, id_to_children_ids, acronym_to_children, acronym_to_all_descendants = \
        build_hierarchy_from_paths(id_to_node)
    
    # ==================== Step 2: 加载Annotation ====================
    logger.info("\nStep 2: 加载Annotation volume...")
    
    annotation_file = data_path / "annotation_25.nrrd"
    logger.info(f"  文件: {annotation_file}")
    
    if not annotation_file.exists():
        logger.error(f"  ✗ Annotation文件不存在: {annotation_file}")
        return None, None
    
    annotation_volume, _ = nrrd.read(str(annotation_file))
    logger.info(f"  ✓ Shape: {annotation_volume.shape}")
    logger.info(f"  ✓ 唯一ID数量: {len(np.unique(annotation_volume))}")
    
    # ==================== Step 3: 加载MERFISH数据 ====================
    logger.info("\nStep 3: 加载MERFISH细胞数据...")
    
    merfish_file = data_path / "merfish_in_ccfv3_with_swc_all.csv"
    logger.info(f"  文件: {merfish_file}")
    
    if not merfish_file.exists():
        logger.error(f"  ✗ MERFISH文件不存在: {merfish_file}")
        return None, None
    
    merfish_df = pd.read_csv(merfish_file)
    logger.info(f"  ✓ 细胞数: {len(merfish_df)}")
    
    # ==================== Step 4: 映射细胞到CCF ====================
    logger.info("\nStep 4: 映射细胞到CCF ID（无聚合）...")
    
    cells_with_ccf = map_cells_to_ccf_simple(merfish_df, annotation_volume)
    cells_with_ccf['ccf_acronym'] = cells_with_ccf['ccf_id'].map(id_to_acronym)
    
    # 统计
    ccf_id_counts = cells_with_ccf['ccf_id'].value_counts()
    acronym_counts = cells_with_ccf['ccf_acronym'].value_counts()
    
    logger.info(f"  ✓ 映射到 {len(ccf_id_counts)} 个不同的CCF ID")
    logger.info(f"  ✓ 映射到 {len(acronym_counts)} 个不同的acronym")
    
    # ==================== Step 5: 检查关键案例 ====================
    logger.info("\nStep 5: 检查关键案例...")
    
    test_cases = [
        'ACAd', 'ACAd2/3', 'ACAd5', 'ACAd6a',
        'ORBl', 'ORBl2/3', 'ORBl5',
        'ENTm', 'ENTm2/3',
        'RSPd', 'RSPd2/3',
    ]
    
    results = []
    
    for acronym in test_cases:
        if acronym not in acronym_to_id:
            logger.warning(f"  ⚠ {acronym} 不在CCF树中")
            continue
        
        ccf_id = acronym_to_id[acronym]
        node = id_to_node[ccf_id]
        
        # 直接映射到该ID的细胞数
        direct_cells = len(cells_with_ccf[cells_with_ccf['ccf_id'] == ccf_id])
        
        # Children信息
        children_acronyms = acronym_to_children.get(acronym, set())
        all_descendants = acronym_to_all_descendants.get(acronym, set())
        
        # Children的细胞数
        children_cell_count = 0
        if children_acronyms:
            children_ids = [acronym_to_id[c] for c in children_acronyms if c in acronym_to_id]
            children_cell_count = len(cells_with_ccf[cells_with_ccf['ccf_id'].isin(children_ids)])
        
        # 记录结果
        result = {
            'acronym': acronym,
            'ccf_id': ccf_id,
            'direct_cells': direct_cells,
            'n_children': len(children_acronyms),
            'n_descendants': len(all_descendants),
            'children_cells': children_cell_count,
            'children_list': list(children_acronyms)[:5] if children_acronyms else []
        }
        results.append(result)
        
        # 打印详细信息
        logger.info(f"\n  {acronym}:")
        logger.info(f"    CCF ID: {ccf_id}")
        logger.info(f"    直接映射的细胞数: {direct_cells}")
        logger.info(f"    Children数量: {len(children_acronyms)}")
        logger.info(f"    所有后代数量: {len(all_descendants)}")
        
        if children_acronyms:
            logger.info(f"    Children样例: {list(children_acronyms)[:5]}")
            logger.info(f"    Children细胞总数: {children_cell_count}")
            
            # 关键诊断
            if direct_cells > 0 and children_cell_count == 0 and len(children_acronyms) > 0:
                logger.warning(f"    ⚠️ 潜在问题：父节点有{direct_cells}个细胞，但所有children都是0！")
                logger.warning(f"       可能原因：")
                logger.warning(f"       1. Annotation volume只标注到父节点级别")
                logger.warning(f"       2. Children被聚合到父节点")
            elif direct_cells > 0 and children_cell_count > 0:
                total = direct_cells + children_cell_count
                parent_ratio = direct_cells / total
                logger.info(f"    父节点占比: {parent_ratio*100:.1f}% ({direct_cells}/{total})")
                if parent_ratio > 0.9:
                    logger.warning(f"    ⚠️ 父节点包含绝大部分细胞")
            elif direct_cells == 0 and children_cell_count > 0:
                logger.info(f"    ✓ 正常：父节点无细胞，细胞分布在children")
            elif direct_cells == 0 and children_cell_count == 0:
                logger.info(f"    该区域没有MERFISH细胞")
    
    # ==================== Step 6: 全局统计 ====================
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: 全局统计分析")
    logger.info("=" * 80)
    
    # 统计有children且自身有细胞的节点
    nodes_with_children_and_cells = 0
    nodes_with_children_no_cells = 0
    nodes_with_children_all_zero = 0
    
    for acronym in acronym_to_id.keys():
        children = acronym_to_children.get(acronym, set())
        if len(children) > 0:  # 有children
            ccf_id = acronym_to_id[acronym]
            direct_cells = len(cells_with_ccf[cells_with_ccf['ccf_id'] == ccf_id])
            
            # Children的细胞数
            children_ids = [acronym_to_id[c] for c in children if c in acronym_to_id]
            children_cells = len(cells_with_ccf[cells_with_ccf['ccf_id'].isin(children_ids)])
            
            if direct_cells > 0:
                nodes_with_children_and_cells += 1
                if children_cells == 0:
                    nodes_with_children_all_zero += 1
            else:
                nodes_with_children_no_cells += 1
    
    logger.info(f"\n整个CCF树的统计:")
    logger.info(f"  - 有children的节点总数: {len([a for a in acronym_to_children if len(acronym_to_children[a]) > 0])}")
    logger.info(f"  - 有children且自身有细胞: {nodes_with_children_and_cells}")
    logger.info(f"  - 有children但自身无细胞: {nodes_with_children_no_cells}")
    logger.info(f"  - 有children，自身有细胞，但所有children都是0: {nodes_with_children_all_zero}")
    
    if nodes_with_children_all_zero > 50:
        logger.warning("\n⚠️⚠️⚠️ 诊断结果：")
        logger.warning(f"  发现 {nodes_with_children_all_zero} 个父节点包含细胞但children全为0！")
        logger.warning("  这强烈暗示其中一个问题：")
        logger.warning("  1. ❌ Annotation volume分辨率太低，只标注到粗粒度")
        logger.warning("  2. ❌ 数据加载代码进行了children聚合")
        logger.warning("  3. ❌ CCF树定义与annotation volume不匹配")
    else:
        logger.info("\n✓ 初步看起来正常，父节点和children的细胞分布合理")
    
    # ==================== Step 7: 保存结果 ====================
    logger.info("\nStep 7: 保存诊断结果...")
    
    results_df = pd.DataFrame(results)
    output_path = Path("/home/claude")
    results_df.to_csv(output_path / "region_aggregation_diagnosis_v2.csv", index=False)
    logger.info(f"  ✓ 结果已保存")
    
    # 保存样例细胞数据
    sample_regions = ['ACAd', 'ACAd2/3', 'ACAd5', 'ORBl', 'ORBl2/3', 'ENTm', 'ENTm2/3']
    sample_cells = cells_with_ccf[cells_with_ccf['ccf_acronym'].isin(sample_regions)]
    if len(sample_cells) > 0:
        sample_cells.head(1000).to_csv(output_path / "sample_cells_v2.csv", index=False)
        logger.info(f"  ✓ 样例细胞数据已保存 ({len(sample_cells)} 个细胞)")
    
    logger.info("\n" + "=" * 80)
    logger.info("诊断完成！")
    logger.info("=" * 80)
    
    return results_df, cells_with_ccf


def main():
    """主函数"""
    data_path = Path("/home/wlj/NeuroXiv2/data")
    
    logger.info(f"数据路径: {data_path}")
    
    try:
        results_df, cells_with_ccf = check_aggregation_behavior(data_path)
        
        if results_df is not None:
            logger.info("\n结论和建议：")
            logger.info("=" * 80)
            logger.info("请查看输出的CSV文件，重点关注：")
            logger.info("1. 哪些区域的父节点包含细胞但children为空")
            logger.info("2. 这种模式是否普遍存在")
            logger.info("3. 是否需要使用ME_Subregion annotation")
            logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
