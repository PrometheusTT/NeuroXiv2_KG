#!/usr/bin/env python3
"""
知识图谱续传插入工具
检测已插入的内容，智能跳过，继续插入剩余部分

功能：
1. 检查每种类型节点的插入情况
2. 标记已完成的步骤
3. 从未完成的步骤继续执行
4. 避免重复插入

作者: wangmajortom
日期: 2025-10-26
"""
import sys
from pathlib import Path
from loguru import logger
from neo4j import GraphDatabase

# 导入主程序的组件（假设在同一目录）
try:
    from KG_ConstructorV4_Neo4j_with_neuron_subregion_subregionrelationship import (
        Neo4jConnector,
        KnowledgeGraphBuilderNeo4j,
        MERFISHHierarchyLoader,
        RegionAnalyzer,
        MorphologyDataLoader,
        NeuronDataLoader,
        SubregionLoader
    )
    from data_loader_enhanced import load_data, prepare_analysis_data
    from neuron_subregion_relationship_inserter import (
        NeuronSubregionRelationshipInserter,
        verify_relationships
    )
except ImportError as e:
    logger.error(f"导入失败: {e}")
    logger.error("请确保脚本在正确的目录中运行")
    sys.exit(1)


class InsertionProgressChecker:
    """插入进度检查器"""

    def __init__(self, neo4j_connector):
        self.neo4j = neo4j_connector
        self.progress = {}

    def check_all_progress(self):
        """检查所有插入进度"""
        logger.info("=" * 60)
        logger.info("检查知识图谱插入进度")
        logger.info("=" * 60)

        checks = [
            ('Region', self.check_regions),
            ('Neuron', self.check_neurons),
            ('Subregion', self.check_subregions),
            ('ME_Subregion', self.check_me_subregions),
            ('Class', self.check_classes),
            ('Subclass', self.check_subclasses),
            ('Supertype', self.check_supertypes),
            ('Cluster', self.check_clusters),
            ('PROJECT_TO', self.check_project_to),
            ('HAS_*', self.check_has_relationships),
            ('BELONGS_TO', self.check_belongs_to),
            ('LOCATE_AT', self.check_locate_at),
            ('DEN/AXON_NEIGHBOURING', self.check_neuron_connections),
            ('LOCATE_AT_SUBREGION', self.check_locate_at_subregion),
            ('LOCATE_AT_ME_SUBREGION', self.check_locate_at_me_subregion),
        ]

        for name, check_func in checks:
            count = check_func()
            self.progress[name] = count
            status = "✓" if count > 0 else "✗"
            logger.info(f"{status} {name}: {count:,}")

        logger.info("=" * 60)
        return self.progress

    def check_regions(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:Region) RETURN count(n) as count")
            return result.single()['count']

    def check_neurons(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:Neuron) RETURN count(n) as count")
            return result.single()['count']

    def check_subregions(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:Subregion) RETURN count(n) as count")
            return result.single()['count']

    def check_me_subregions(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:ME_Subregion) RETURN count(n) as count")
            return result.single()['count']

    def check_classes(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:Class) RETURN count(n) as count")
            return result.single()['count']

    def check_subclasses(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:Subclass) RETURN count(n) as count")
            return result.single()['count']

    def check_supertypes(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:Supertype) RETURN count(n) as count")
            return result.single()['count']

    def check_clusters(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH (n:Cluster) RETURN count(n) as count")
            return result.single()['count']

    def check_project_to(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH ()-[r:PROJECT_TO]->() RETURN count(r) as count")
            return result.single()['count']

    def check_has_relationships(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("""
                MATCH ()-[r]->() 
                WHERE type(r) IN ['HAS_CLASS', 'HAS_SUBCLASS', 'HAS_SUPERTYPE', 'HAS_CLUSTER']
                RETURN count(r) as count
            """)
            return result.single()['count']

    def check_belongs_to(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as count")
            return result.single()['count']

    def check_locate_at(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH ()-[r:LOCATE_AT]->() RETURN count(r) as count")
            return result.single()['count']

    def check_neuron_connections(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("""
                MATCH ()-[r]->() 
                WHERE type(r) IN ['DEN_NEIGHBOURING', 'AXON_NEIGHBOURING']
                RETURN count(r) as count
            """)
            return result.single()['count']

    def check_locate_at_subregion(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH ()-[r:LOCATE_AT_SUBREGION]->() RETURN count(r) as count")
            return result.single()['count']

    def check_locate_at_me_subregion(self):
        with self.neo4j.driver.session(database=self.neo4j.database) as session:
            result = session.run("MATCH ()-[r:LOCATE_AT_ME_SUBREGION]->() RETURN count(r) as count")
            return result.single()['count']

    def get_missing_steps(self):
        """获取需要执行的步骤"""
        missing = []

        # 定义各步骤的预期最小值
        expected = {
            'Region': 100,
            'Neuron': 100000,
            'Subregion': 300,
            'ME_Subregion': 600,
            'Class': 5,
            'Subclass': 20,
            'Supertype': 100,
            'Cluster': 500,
            'PROJECT_TO': 1000,
            'HAS_*': 1000,
            'BELONGS_TO': 500,
            'LOCATE_AT': 100000,
            'DEN/AXON_NEIGHBOURING': 10000,
            'LOCATE_AT_SUBREGION': 50000,
            'LOCATE_AT_ME_SUBREGION': 70000,
        }

        for step, min_expected in expected.items():
            actual = self.progress.get(step, 0)
            if actual < min_expected:
                missing.append({
                    'step': step,
                    'expected': min_expected,
                    'actual': actual,
                    'missing': min_expected - actual
                })

        return missing


def resume_insertion(data_dir, neo4j_conn, database_name, skip_completed=True):
    """
    续传插入

    参数:
        data_dir: 数据目录
        neo4j_conn: Neo4j连接器
        database_name: 数据库名称
        skip_completed: 是否跳过已完成的步骤
    """
    logger.info("=" * 60)
    logger.info("开始续传插入")
    logger.info("=" * 60)

    # 1. 检查进度
    checker = InsertionProgressChecker(neo4j_conn)
    progress = checker.check_all_progress()

    if skip_completed:
        missing = checker.get_missing_steps()

        if not missing:
            logger.info("\n✓ 所有步骤已完成！")
            return

        logger.info("\n需要执行的步骤:")
        for item in missing:
            logger.info(
                f"  - {item['step']}: 已有 {item['actual']:,}, 预期 {item['expected']:,}, 缺少 {item['missing']:,}")

    # 2. 加载数据（如果需要）
    data_path = Path(data_dir)

    logger.info("\n" + "=" * 60)
    logger.info("加载基础数据")
    logger.info("=" * 60)

    data = load_data(data_path)
    processed_data = prepare_analysis_data(data)

    region_data = processed_data.get('region_data', None)
    merfish_cells = processed_data.get('merfish_cells', None)
    projection_data = processed_data.get('projection_df', None)
    tree_data = processed_data.get('tree', [])

    # 创建builder
    builder = KnowledgeGraphBuilderNeo4j(neo4j_conn)

    # 初始化区域分析器
    if tree_data:
        builder.region_analyzer = RegionAnalyzer(tree_data)
        logger.info(f"✓ 区域分析器已初始化")

    # 加载层级数据
    hierarchy_loader = MERFISHHierarchyLoader(data_path / "hierarchy.json")
    if hierarchy_loader.load_hierarchy():
        builder.set_hierarchy_loader(hierarchy_loader)
        logger.info(f"✓ 层级数据已加载")

    # 加载形态学数据
    morphology_loader = MorphologyDataLoader(data_path, builder.region_analyzer)
    if morphology_loader.load_morphology_data():
        if projection_data is not None:
            morphology_loader.set_projection_data(projection_data)
        builder.morphology_loader = morphology_loader
        logger.info(f"✓ 形态学数据已加载")

    # 加载神经元数据
    neuron_loader = None
    if progress.get('Neuron', 0) < 100000 or progress.get('LOCATE_AT', 0) < 100000:
        neuron_loader = NeuronDataLoader(data_path, builder.region_analyzer, builder.morphology_loader)
        if neuron_loader.load_neuron_data():
            neuron_loader.process_neuron_data()
            logger.info(f"✓ 神经元数据已加载: {len(neuron_loader.neurons_data):,} 个")

    # 加载Subregion数据
    subregion_loader = None
    if progress.get('Subregion', 0) < 300 or progress.get('ME_Subregion', 0) < 600:
        ccf_me_path = data_path / "surf_tree_ccf-me.json"
        subregion_loader = SubregionLoader(ccf_me_path)
        if subregion_loader.load_subregion_data():
            logger.info(f"✓ Subregion数据已加载")

    # 3. 按需执行插入步骤
    logger.info("\n" + "=" * 60)
    logger.info("执行续传插入")
    logger.info("=" * 60)

    # Step 1: Region节点
    if not skip_completed or progress.get('Region', 0) < 100:
        logger.info("\n插入 Region 节点...")
        builder.generate_and_insert_unified_region_nodes(region_data, merfish_cells)
    else:
        logger.info("\n✓ 跳过 Region 节点（已存在）")

    # Step 2: Neuron节点
    if not skip_completed or progress.get('Neuron', 0) < 100000:
        if neuron_loader:
            logger.info("\n插入 Neuron 节点...")
            builder.generate_and_insert_neuron_nodes(neuron_loader)
        else:
            logger.warning("\n⚠ Neuron数据未加载，跳过")
    else:
        logger.info("\n✓ 跳过 Neuron 节点（已存在）")

    # Step 3: Subregion节点
    if not skip_completed or progress.get('Subregion', 0) < 300:
        if subregion_loader:
            logger.info("\n插入 Subregion 节点...")
            builder.generate_and_insert_subregion_nodes(subregion_loader)
        else:
            logger.warning("\n⚠ Subregion数据未加载，跳过")
    else:
        logger.info("\n✓ 跳过 Subregion 节点（已存在）")

    # Step 4: ME_Subregion节点
    if not skip_completed or progress.get('ME_Subregion', 0) < 600:
        if subregion_loader:
            logger.info("\n插入 ME_Subregion 节点...")
            builder.generate_and_insert_me_subregion_nodes(subregion_loader)
        else:
            logger.warning("\n⚠ ME_Subregion数据未加载，跳过")
    else:
        logger.info("\n✓ 跳过 ME_Subregion 节点（已存在）")

    # Step 5: MERFISH细胞类型节点
    if not skip_completed or any([
        progress.get('Class', 0) < 5,
        progress.get('Subclass', 0) < 20,
        progress.get('Supertype', 0) < 100,
        progress.get('Cluster', 0) < 500
    ]):
        logger.info("\n插入 MERFISH 细胞类型节点...")
        builder.generate_and_insert_merfish_nodes_from_hierarchy(merfish_cells)
    else:
        logger.info("\n✓ 跳过 MERFISH 细胞类型节点（已存在）")

    # Step 6: HAS关系
    if not skip_completed or progress.get('HAS_*', 0) < 1000:
        logger.info("\n插入 HAS_* 关系...")
        for level in ['class', 'subclass', 'supertype', 'cluster']:
            builder.generate_and_insert_has_relationships_unified(merfish_cells, level)
    else:
        logger.info("\n✓ 跳过 HAS_* 关系（已存在）")

    # Step 7: BELONGS_TO关系
    if not skip_completed or progress.get('BELONGS_TO', 0) < 500:
        logger.info("\n插入 BELONGS_TO 关系...")
        builder.generate_and_insert_belongs_to_from_hierarchy()
        if subregion_loader:
            builder.generate_and_insert_subregion_relationships(subregion_loader)
    else:
        logger.info("\n✓ 跳过 BELONGS_TO 关系（已存在）")

    # Step 8: PROJECT_TO关系
    if not skip_completed or progress.get('PROJECT_TO', 0) < 1000:
        logger.info("\n插入 PROJECT_TO 关系...")
        builder.generate_and_insert_project_to_relationships(projection_data)
    else:
        logger.info("\n✓ 跳过 PROJECT_TO 关系（已存在）")

    # Step 9: 神经元关系
    if not skip_completed or any([
        progress.get('LOCATE_AT', 0) < 100000,
        progress.get('DEN/AXON_NEIGHBOURING', 0) < 10000
    ]):
        if neuron_loader:
            logger.info("\n插入神经元相关关系...")
            builder.generate_and_insert_neuron_relationships(neuron_loader)
        else:
            logger.warning("\n⚠ Neuron数据未加载，跳过神经元关系")
    else:
        logger.info("\n✓ 跳过神经元关系（已存在）")

    # Step 10: 神经元-Subregion关系
    if not skip_completed or any([
        progress.get('LOCATE_AT_SUBREGION', 0) < 50000,
        progress.get('LOCATE_AT_ME_SUBREGION', 0) < 70000
    ]):
        logger.info("\n插入神经元-Subregion关系...")
        neuron_subregion_inserter = NeuronSubregionRelationshipInserter(neo4j_conn, data_path)

        if neuron_subregion_inserter.load_neuron_subregion_mapping():
            neuron_subregion_inserter.insert_all_relationships(batch_size=1000)
            verify_relationships(neo4j_conn, database_name)
        else:
            logger.warning("\n⚠ 无法加载神经元-Subregion映射")
    else:
        logger.info("\n✓ 跳过神经元-Subregion关系（已存在）")

    # 最终检查
    logger.info("\n" + "=" * 60)
    logger.info("续传完成，最终检查")
    logger.info("=" * 60)
    final_progress = checker.check_all_progress()

    logger.info("\n✓ 续传插入完成！")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='知识图谱续传插入工具')
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--neo4j_password', type=str, required=True)
    parser.add_argument('--database', type=str, default='neo4j')
    parser.add_argument('--check_only', action='store_true',
                        help='仅检查进度，不执行插入')
    parser.add_argument('--force_all', action='store_true',
                        help='强制执行所有步骤（不跳过）')

    args = parser.parse_args()

    # 连接Neo4j
    neo4j_conn = Neo4jConnector(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.database
    )

    if not neo4j_conn.connect():
        logger.error("无法连接到Neo4j")
        return 1

    try:
        if args.check_only:
            # 仅检查进度
            checker = InsertionProgressChecker(neo4j_conn)
            progress = checker.check_all_progress()
            missing = checker.get_missing_steps()

            if missing:
                logger.info("\n缺失的步骤:")
                for item in missing:
                    logger.info(f"  {item['step']}: 缺少 {item['missing']:,}")
            else:
                logger.info("\n✓ 所有步骤已完成！")
        else:
            # 执行续传插入
            resume_insertion(
                args.data_dir,
                neo4j_conn,
                args.database,
                skip_completed=not args.force_all
            )

        return 0

    except Exception as e:
        logger.error(f"续传插入失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    sys.exit(main())