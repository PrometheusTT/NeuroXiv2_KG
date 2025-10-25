#!/usr/bin/env python3
"""
测试神经元ME_Subregion映射 V2 (修正版)

关键修正:
1. ME_Subregion必须带-ME后缀
2. 使用info.csv中的celltype作为Subregion（不重新查找）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Neuron_MESubregion_Mapper import NeuronMESubregionMapperV2
from loguru import logger


def test_mapper_v2():
    """测试V2映射器"""

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("test_mapping_v2.log", rotation="100 MB", level="DEBUG")

    logger.info("=" * 60)
    logger.info("测试神经元ME_Subregion映射 V2 (修正版)")
    logger.info("=" * 60)

    # 文件路径
    nrrd_file = Path("/home/wlj/NeuroXiv2/data/parc_r671_full.nrrd")
    pkl_file = Path("/home/wlj/NeuroXiv2/data/parc_r671_full.nrrd.pkl")

    data_dir = Path("/home/wlj/NeuroXiv2/data")
    soma_file = data_dir / "soma.csv"
    info_file = data_dir / "info.csv"
    json_file = data_dir / "surf_tree_ccf-me.json"

    output_file = Path("/home/wlj/NeuroXiv2/data/info_with_me_subregion_v2.csv")

    # 检查文件
    logger.info("检查文件存在性...")
    required_files = {
        "NRRD": nrrd_file,
        "PKL": pkl_file,
        "Soma": soma_file,
        "Info": info_file
    }

    all_exist = True
    for name, path in required_files.items():
        if path.exists():
            logger.info(f"✓ {name}文件: {path}")
        else:
            logger.error(f"✗ {name}文件不存在: {path}")
            all_exist = False

    if not all_exist:
        logger.error("缺少必需文件，请检查文件路径")
        return False

    # JSON文件是可选的
    if json_file.exists():
        logger.info(f"✓ JSON文件: {json_file}")
    else:
        logger.warning(f"⚠ JSON文件不存在: {json_file}")
        logger.warning("  将使用简单格式显示区域名称")
        json_file = None

    logger.info("=" * 60)

    # 创建映射器
    logger.info("创建V2映射器...")
    mapper = NeuronMESubregionMapperV2(
        nrrd_file=nrrd_file,
        pkl_file=pkl_file,
        soma_file=soma_file,
        info_file=info_file,
        json_tree_file=json_file
    )

    # 运行映射
    logger.info("开始运行映射流程...")
    success = mapper.run_full_pipeline(output_info_file=output_file)

    if success:
        logger.info("=" * 60)
        logger.info("✓ 映射成功完成!")
        logger.info(f"✓ 输出文件: {output_file}")
        logger.info("=" * 60)

        # 读取并分析结果
        import pandas as pd
        result_df = pd.read_csv(output_file)

        # 统计
        total = len(result_df)
        has_me = result_df['me_subregion_id'].notna().sum()
        has_subregion = result_df['subregion'].notna().sum()

        logger.info("结果统计:")
        logger.info(f"  总神经元数: {total}")
        logger.info(f"  有Subregion信息: {has_subregion} ({has_subregion / total * 100:.1f}%)")
        logger.info(f"  有ME_Subregion信息: {has_me} ({has_me / total * 100:.1f}%)")

        # 显示ME_Subregion样例
        me_samples = result_df[result_df['is_me_subregion'] == True].head(10)
        if not me_samples.empty:
            logger.info("\nME_Subregion映射样例（前10个）:")
            for idx, row in me_samples.iterrows():
                logger.info(f"  {row['ID']}: "
                            f"Subregion={row['subregion']}, "
                            f"ME={row['me_subregion_acronym']}")

        # 验证ME后缀
        if has_me > 0:
            me_acronyms = result_df[result_df['me_subregion_acronym'].notna()]['me_subregion_acronym']
            me_with_suffix = me_acronyms.str.endswith('-ME').sum()
            logger.info(f"\n✓ 验证: {me_with_suffix}/{has_me} 个ME_Subregion带有-ME后缀")

            if me_with_suffix < has_me:
                logger.warning(f"⚠ 警告: 有 {has_me - me_with_suffix} 个ME_Subregion不带-ME后缀!")
                non_me_samples = result_df[
                    result_df['me_subregion_acronym'].notna() &
                    ~result_df['me_subregion_acronym'].str.endswith('-ME')
                    ]['me_subregion_acronym'].head(5)
                logger.warning(f"  样例: {non_me_samples.tolist()}")

        logger.info("=" * 60)
    else:
        logger.error("✗ 映射失败!")
        return False

    return True


if __name__ == "__main__":
    success = test_mapper_v2()
    sys.exit(0 if success else 1)