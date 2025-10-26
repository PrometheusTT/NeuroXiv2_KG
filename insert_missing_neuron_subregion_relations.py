#!/usr/bin/env python3
"""
直接插入缺失的神经元-Subregion关系

只插入两个缺失的关系，不检查其他内容

使用方法:
    python insert_missing_neuron_subregion_relations.py --neo4j_password neuroxiv
"""
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='插入缺失的神经元-Subregion关系')
    parser.add_argument('--neo4j_password', type=str, required=True)
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--database', type=str, default='neo4j')
    parser.add_argument('--data_dir', type=str, default='/home/wlj/NeuroXiv2/data')
    parser.add_argument('--batch_size', type=int, default=1000)

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("插入缺失的神经元-Subregion关系")
    logger.info("=" * 60)

    # 导入必要的模块
    try:
        from neuron_subregion_relationship_inserter import (
            NeuronSubregionRelationshipInserter,
            verify_relationships
        )
        # 尝试两种可能的导入路径
        try:
            from KG_ConstructorV4_Neo4j_with_neuron_subregion_subregionrelationship import Neo4jConnector
        except ImportError:
            from KG_ConstructorV4_Neo4j_with_neuron_subregion import Neo4jConnector
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.error("请确保在正确的目录运行此脚本")
        return 1

    # 连接Neo4j
    logger.info(f"连接到Neo4j: {args.neo4j_uri}/{args.database}")
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
        # 先检查当前状态
        logger.info("\n检查当前关系状态...")
        with neo4j_conn.driver.session(database=args.database) as session:
            result1 = session.run("MATCH ()-[r:LOCATE_AT_SUBREGION]->() RETURN count(r) as count")
            count1 = result1.single()['count']

            result2 = session.run("MATCH ()-[r:LOCATE_AT_ME_SUBREGION]->() RETURN count(r) as count")
            count2 = result2.single()['count']

            logger.info(f"LOCATE_AT_SUBREGION: {count1:,}")
            logger.info(f"LOCATE_AT_ME_SUBREGION: {count2:,}")

            if count1 > 0 and count2 > 0:
                logger.info("\n✓ 关系已存在，无需重复插入")
                response = input("\n是否强制重新插入? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("取消操作")
                    return 0

        # 创建插入器
        logger.info("\n创建关系插入器...")
        data_path = Path(args.data_dir)
        inserter = NeuronSubregionRelationshipInserter(neo4j_conn, data_path)

        # 加载映射数据
        logger.info("加载神经元-Subregion映射数据...")
        if not inserter.load_neuron_subregion_mapping():
            logger.error("加载映射数据失败")
            return 1

        # 插入关系
        logger.info("\n" + "=" * 60)
        logger.info("开始插入关系")
        logger.info("=" * 60)
        inserter.insert_all_relationships(batch_size=args.batch_size)

        # 验证
        logger.info("\n" + "=" * 60)
        logger.info("验证插入结果")
        logger.info("=" * 60)
        verify_relationships(neo4j_conn, args.database)

        logger.info("\n" + "=" * 60)
        logger.info("✓ 完成！")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"插入失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    sys.exit(main())