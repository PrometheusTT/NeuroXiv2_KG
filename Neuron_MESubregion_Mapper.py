"""
神经元ME_Subregion映射工具 V2 - 修正版
基于已知的Subregion（celltype）和soma坐标精确定位ME_Subregion

关键修正：
1. ME_Subregion带有-ME后缀
2. 使用info.csv中已有的celltype作为Subregion
3. 通过soma坐标在NRRD中查找ME_Subregion（带-ME后缀的区域）

作者: Claude (修正版)
日期: 2025-10-25
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger
import nrrd


class NeuronMESubregionMapperV2:
    """修正版：将神经元映射到ME_Subregion（带-ME后缀）"""

    def __init__(self,
                 nrrd_file: Path,
                 pkl_file: Path,
                 soma_file: Path,
                 info_file: Path,
                 json_tree_file: Optional[Path] = None):
        """
        初始化映射器

        参数:
            nrrd_file: ME注释的nrrd文件（包含ME_Subregion）
            pkl_file: NRRD value到区域ID的映射pkl文件
            soma_file: 包含神经元soma坐标的CSV文件
            info_file: 神经元信息文件（包含celltype=Subregion）
            json_tree_file: 区域层级树JSON文件（用于获取区域名称）
        """
        self.nrrd_file = nrrd_file
        self.pkl_file = pkl_file
        self.soma_file = soma_file
        self.info_file = info_file
        self.json_tree_file = json_tree_file

        # 数据容器
        self.annotation_volume = None
        self.annotation_header = None
        self.nrrd_to_region_id_map = None  # NRRD value -> Region ID
        self.region_id_to_info_map = {}    # Region ID -> 区域信息（包括名称）
        self.soma_df = None
        self.info_df = None

        # 统计信息
        self.stats = {
            'total_neurons': 0,
            'mapped_to_me_subregion': 0,
            'mapped_to_subregion_only': 0,
            'unmapped': 0
        }

    def load_nrrd_annotation(self) -> bool:
        """加载NRRD注释文件"""
        logger.info(f"加载NRRD注释文件: {self.nrrd_file}")

        try:
            self.annotation_volume, self.annotation_header = nrrd.read(str(self.nrrd_file))
            logger.info(f"NRRD体积形状: {self.annotation_volume.shape}")
            logger.info(f"NRRD数据类型: {self.annotation_volume.dtype}")

            # 检查注释值的范围
            unique_ids = np.unique(self.annotation_volume)
            logger.info(f"注释中包含 {len(unique_ids)} 个唯一值")
            logger.info(f"值范围: {unique_ids.min()} - {unique_ids.max()}")

            return True

        except Exception as e:
            logger.error(f"加载NRRD文件失败: {e}")
            return False

    def load_pkl_mapping(self) -> bool:
        """
        加载PKL映射文件

        PKL格式: {nrrd_voxel_value: region_id}
        """
        logger.info(f"加载PKL映射文件: {self.pkl_file}")

        try:
            with open(self.pkl_file, 'rb') as f:
                self.nrrd_to_region_id_map = pickle.load(f)

            logger.info(f"PKL文件类型: {type(self.nrrd_to_region_id_map)}")

            if not isinstance(self.nrrd_to_region_id_map, dict):
                logger.error(f"PKL文件不是字典格式")
                return False

            logger.info(f"加载了 {len(self.nrrd_to_region_id_map)} 个NRRD->Region ID映射")

            # 显示样例
            sample_items = list(self.nrrd_to_region_id_map.items())[:5]
            logger.info(f"映射样例 (NRRD value -> Region ID):")
            for nrrd_val, region_id in sample_items:
                logger.info(f"  {nrrd_val} -> {region_id}")

            return True

        except Exception as e:
            logger.error(f"加载PKL文件失败: {e}")
            return False

    def load_region_info_from_json(self) -> bool:
        """从JSON文件加载区域信息"""
        if not self.json_tree_file or not self.json_tree_file.exists():
            logger.warning(f"JSON文件不存在或未提供: {self.json_tree_file}")
            return False

        logger.info(f"加载区域信息: {self.json_tree_file}")

        try:
            import json
            with open(self.json_tree_file, 'r') as f:
                tree_data = json.load(f)

            # 递归提取所有区域信息
            self._extract_region_info_from_tree(tree_data)

            logger.info(f"从JSON加载了 {len(self.region_id_to_info_map)} 个区域信息")

            # 统计ME_Subregion数量
            me_count = sum(1 for info in self.region_id_to_info_map.values()
                          if info.get('acronym', '').endswith('-ME'))
            logger.info(f"其中包含 {me_count} 个ME_Subregion（带-ME后缀）")

            # 显示一些ME_Subregion样例
            me_samples = [info for info in self.region_id_to_info_map.values()
                         if info.get('acronym', '').endswith('-ME')][:5]
            if me_samples:
                logger.info("ME_Subregion样例:")
                for info in me_samples:
                    logger.info(f"  ID {info['id']}: {info['acronym']} ({info.get('name', 'N/A')})")

            return True

        except Exception as e:
            logger.error(f"加载JSON文件失败: {e}")
            return False

    def _extract_region_info_from_tree(self, nodes, parent_info=None):
        """递归提取区域信息"""
        if isinstance(nodes, dict):
            nodes = [nodes]

        for node in nodes:
            if not isinstance(node, dict):
                continue

            # 提取节点信息
            node_id = node.get('id')
            acronym = node.get('acronym', '')
            name = node.get('name', '')

            if node_id is not None:
                self.region_id_to_info_map[node_id] = {
                    'id': node_id,
                    'acronym': acronym,
                    'name': name,
                    'is_me_subregion': acronym.endswith('-ME'),
                    'parent': parent_info
                }

            # 递归处理子节点
            children = node.get('children', [])
            if children:
                parent_for_children = {'id': node_id, 'acronym': acronym}
                self._extract_region_info_from_tree(children, parent_for_children)

    def load_soma_coordinates(self) -> bool:
        """加载soma坐标数据"""
        logger.info(f"加载soma坐标文件: {self.soma_file}")

        try:
            self.soma_df = pd.read_csv(self.soma_file)
            logger.info(f"加载了 {len(self.soma_df)} 条soma记录")

            # 检查必要的列
            required_cols = ['ID', 'x', 'y', 'z']
            missing_cols = [col for col in required_cols if col not in self.soma_df.columns]

            if missing_cols:
                logger.error(f"Soma文件缺少必要的列: {missing_cols}")
                logger.info(f"可用列: {list(self.soma_df.columns)}")
                return False

            # 显示坐标范围
            logger.info(f"X坐标范围: {self.soma_df['x'].min():.1f} - {self.soma_df['x'].max():.1f}")
            logger.info(f"Y坐标范围: {self.soma_df['y'].min():.1f} - {self.soma_df['y'].max():.1f}")
            logger.info(f"Z坐标范围: {self.soma_df['z'].min():.1f} - {self.soma_df['z'].max():.1f}")

            return True

        except Exception as e:
            logger.error(f"加载soma文件失败: {e}")
            return False

    def load_info_file(self) -> bool:
        """加载info.csv文件"""
        logger.info(f"加载info文件: {self.info_file}")

        try:
            self.info_df = pd.read_csv(self.info_file)
            logger.info(f"加载了 {len(self.info_df)} 条info记录")
            logger.info(f"Info列名: {list(self.info_df.columns)}")

            # 检查celltype列
            if 'celltype' in self.info_df.columns:
                non_null_celltype = self.info_df['celltype'].notna().sum()
                logger.info(f"包含 {non_null_celltype}/{len(self.info_df)} 个有celltype的记录")

                # 显示celltype样例
                sample_celltypes = self.info_df['celltype'].dropna().head(5).tolist()
                logger.info(f"Celltype样例: {sample_celltypes}")
            else:
                logger.warning("Info文件中没有celltype列")

            return True

        except Exception as e:
            logger.error(f"加载info文件失败: {e}")
            return False

    def coordinate_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        将物理坐标转换为体素索引

        假设坐标是CCF空间的微米坐标，分辨率为25微米
        """
        resolution = 25.0

        voxel_x = int(x / resolution)
        voxel_y = int(y / resolution)
        voxel_z = int(z / resolution)

        return voxel_x, voxel_y, voxel_z

    def get_region_at_coordinate(self, x: float, y: float, z: float) -> Tuple[Optional[int], Optional[str], bool]:
        """
        获取指定坐标处的区域信息

        返回:
            (region_id, acronym, is_me_subregion)
            is_me_subregion: True表示该区域是ME_Subregion（带-ME后缀）
        """
        # 转换为体素索引
        voxel_x, voxel_y, voxel_z = self.coordinate_to_voxel(x, y, z)

        # 检查边界
        if not (0 <= voxel_x < self.annotation_volume.shape[0] and
                0 <= voxel_y < self.annotation_volume.shape[1] and
                0 <= voxel_z < self.annotation_volume.shape[2]):
            return None, None, False

        # 获取NRRD体素值
        nrrd_value = int(self.annotation_volume[voxel_x, voxel_y, voxel_z])

        if nrrd_value == 0:
            return None, None, False

        # 通过PKL映射获取region ID
        region_id = self.nrrd_to_region_id_map.get(nrrd_value)

        if region_id is None:
            return None, None, False

        # 获取区域信息
        region_info = self.region_id_to_info_map.get(region_id)

        if region_info:
            acronym = region_info['acronym']
            is_me = region_info['is_me_subregion']
            return region_id, acronym, is_me
        else:
            # 如果没有区域信息，假设非ME
            return region_id, f"Region_{region_id}", False

    def map_neurons_to_regions(self) -> pd.DataFrame:
        """
        将所有神经元映射到区域

        对于每个神经元：
        1. 从info.csv获取已知的Subregion（celltype）
        2. 通过soma坐标查找NRRD中的区域
        3. 如果找到的是ME_Subregion（带-ME后缀），则记录
        4. 否则只记录Subregion

        返回包含以下列的DataFrame:
        - ID: 神经元ID
        - subregion: 从celltype获取的Subregion
        - me_subregion_id: 通过soma坐标找到的ME_Subregion ID（如果有）
        - me_subregion_acronym: ME_Subregion的缩写（如果有）
        - found_region_id: NRRD中找到的区域ID
        - found_region_acronym: NRRD中找到的区域缩写
        - is_me_subregion: 找到的区域是否为ME_Subregion
        """
        logger.info("开始映射神经元到区域...")

        # 合并info和soma数据
        merged_df = pd.merge(
            self.info_df[['ID', 'celltype']],
            self.soma_df[['ID', 'x', 'y', 'z']],
            on='ID',
            how='inner'
        )

        logger.info(f"合并后有 {len(merged_df)} 个神经元有完整信息")

        results = []
        self.stats['total_neurons'] = len(merged_df)

        for idx, row in merged_df.iterrows():
            neuron_id = row['ID']
            celltype = row.get('celltype', '')
            x, y, z = row['x'], row['y'], row['z']

            # 通过soma坐标查找区域
            region_id, acronym, is_me = self.get_region_at_coordinate(x, y, z)

            result = {
                'ID': neuron_id,
                'subregion': celltype if pd.notna(celltype) else None,
                'soma_x': x,
                'soma_y': y,
                'soma_z': z,
                'found_region_id': region_id,
                'found_region_acronym': acronym,
                'is_me_subregion': is_me
            }

            # 如果找到的是ME_Subregion，记录详细信息
            if is_me:
                result['me_subregion_id'] = region_id
                result['me_subregion_acronym'] = acronym
                self.stats['mapped_to_me_subregion'] += 1
            else:
                result['me_subregion_id'] = None
                result['me_subregion_acronym'] = None
                if region_id:
                    self.stats['mapped_to_subregion_only'] += 1
                else:
                    self.stats['unmapped'] += 1

            results.append(result)

            # 定期报告进度
            if (idx + 1) % 1000 == 0:
                logger.info(f"已处理 {idx + 1}/{len(merged_df)} 个神经元")

        # 打印统计信息
        logger.info("="*60)
        logger.info("映射统计:")
        logger.info(f"  总神经元数: {self.stats['total_neurons']}")
        logger.info(f"  映射到ME_Subregion: {self.stats['mapped_to_me_subregion']} "
                   f"({self.stats['mapped_to_me_subregion']/self.stats['total_neurons']*100:.1f}%)")
        logger.info(f"  仅映射到Subregion: {self.stats['mapped_to_subregion_only']} "
                   f"({self.stats['mapped_to_subregion_only']/self.stats['total_neurons']*100:.1f}%)")
        logger.info(f"  未映射: {self.stats['unmapped']} "
                   f"({self.stats['unmapped']/self.stats['total_neurons']*100:.1f}%)")
        logger.info("="*60)

        # 显示映射样例
        result_df = pd.DataFrame(results)
        me_mapped = result_df[result_df['is_me_subregion'] == True]

        if not me_mapped.empty:
            logger.info("ME_Subregion映射样例:")
            for idx, row in me_mapped.head(5).iterrows():
                logger.info(f"  神经元 {row['ID']}: "
                          f"Subregion={row['subregion']}, "
                          f"ME_Subregion={row['me_subregion_acronym']}")

        return result_df

    def update_info_file(self, mapping_df: pd.DataFrame, output_file: Optional[Path] = None):
        """
        更新info.csv文件，添加ME_Subregion信息

        参数:
            mapping_df: 包含映射结果的DataFrame
            output_file: 输出文件路径，如果为None则覆盖原文件
        """
        logger.info("更新info文件...")

        # 选择要添加的列
        columns_to_add = [
            'ID',
            'me_subregion_id',
            'me_subregion_acronym',
            'found_region_id',
            'found_region_acronym',
            'is_me_subregion'
        ]

        # 合并到原info_df
        updated_info = self.info_df.merge(
            mapping_df[columns_to_add],
            on='ID',
            how='left'
        )

        # 确定输出文件
        if output_file is None:
            output_file = self.info_file

        # 保存
        updated_info.to_csv(output_file, index=False)
        logger.info(f"已保存更新后的info文件到: {output_file}")

        # 统计
        me_count = updated_info['me_subregion_id'].notna().sum()
        logger.info(f"Info文件中 {me_count}/{len(updated_info)} 个神经元有ME_Subregion信息")

    def run_full_pipeline(self, output_info_file: Optional[Path] = None) -> bool:
        """运行完整的映射流程"""
        logger.info("="*60)
        logger.info("开始神经元区域映射流程 V2（修正版）")
        logger.info("="*60)

        # 1. 加载NRRD注释
        if not self.load_nrrd_annotation():
            return False

        # 2. 加载PKL映射
        if not self.load_pkl_mapping():
            return False

        # 3. 加载区域信息（可选但推荐）
        self.load_region_info_from_json()

        # 4. 加载soma坐标
        if not self.load_soma_coordinates():
            return False

        # 5. 加载info文件
        if not self.load_info_file():
            return False

        # 6. 执行映射
        mapping_df = self.map_neurons_to_regions()

        # 7. 更新info文件
        self.update_info_file(mapping_df, output_info_file)

        logger.info("="*60)
        logger.info("映射流程完成")
        logger.info("="*60)

        return True


def main():
    """主函数"""
    from loguru import logger
    import sys

    # 设置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("neuron_me_mapping_v2.log", rotation="100 MB", level="DEBUG")

    # 文件路径
    data_dir = Path("../data")

    nrrd_file = Path("/mnt/user-data/uploads/parc_r671_full.nrrd")
    pkl_file = Path("/mnt/user-data/uploads/parc_r671_full_nrrd.pkl")
    soma_file = data_dir / "soma.csv"
    info_file = data_dir / "info.csv"
    json_file = data_dir / "surf_tree_ccf-me.json"
    output_file = data_dir / "info_with_me_subregion_v2.csv"

    # 检查必需文件
    required_files = [nrrd_file, pkl_file, soma_file, info_file]
    for f in required_files:
        if not f.exists():
            logger.error(f"必需文件不存在: {f}")
            return False

    # JSON文件是可选的
    if not json_file.exists():
        logger.warning(f"JSON文件不存在: {json_file}")
        logger.warning("将使用简单的Region ID作为名称")
        json_file = None

    # 创建映射器
    mapper = NeuronMESubregionMapperV2(
        nrrd_file=nrrd_file,
        pkl_file=pkl_file,
        soma_file=soma_file,
        info_file=info_file,
        json_tree_file=json_file
    )

    # 运行映射流程
    success = mapper.run_full_pipeline(output_info_file=output_file)

    if success:
        logger.info("✓ 映射成功完成!")
        logger.info(f"✓ 更新后的info文件: {output_file}")
    else:
        logger.error("✗ 映射失败!")

    return success


if __name__ == "__main__":
    main()