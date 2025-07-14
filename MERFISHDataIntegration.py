import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pickle
from pathlib import Path
import tempfile
import os
import argparse
from collections import defaultdict
import time
import sys
import csv
import re
import h5py
from scipy import stats

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MERFISHDataIntegration:
    """MERFISH数据集成类 - 用于计算细胞类型在各RegionLayer中的分布"""

    def __init__(self, cache_dir: Optional[str] = None, include_job_id: bool = True,
                 output_dir: Optional[str] = None):
        """
        初始化MERFISH数据集成器

        Args:
            cache_dir: 缓存目录路径，默认使用系统临时目录
            include_job_id: 是否在缓存文件名中包含作业ID，避免冲突
            output_dir: 输出CSV文件的目录
        """
        # 如果未指定缓存目录，使用系统临时目录
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "merfish_cache"
        else:
            self.cache_dir = Path(cache_dir)

        # 输出目录
        if output_dir is None:
            self.output_dir = Path("merfish_output")
        else:
            self.output_dir = Path(output_dir)

        # 创建缓存和输出目录
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 如果需要，在缓存文件名中包含作业ID
        self.job_id = ""
        if include_job_id:
            self.job_id = f"_{os.getpid()}"

        logger.info(f"使用缓存目录: {self.cache_dir}")
        logger.info(f"使用输出目录: {self.output_dir}")

        # 尝试初始化AllenSDK 或 ABC Atlas
        self.allen_api = None
        self.abc_cache = None

        try:
            # 尝试导入AllenSDK
            from allensdk.api.queries.cell_types_api import CellTypesApi
            self.allen_api = CellTypesApi()
            logger.info("成功初始化AllenSDK CellTypesApi")
        except ImportError:
            logger.warning("无法导入AllenSDK，尝试ABC Atlas...")
            try:
                # 尝试导入ABC Atlas
                from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
                self.abc_cache = AbcProjectCache.from_cache_dir(
                    cache_dir=self.cache_dir / f"abc_cache{self.job_id}"
                )
                logger.info("成功初始化ABC Atlas缓存")
            except ImportError:
                logger.warning("无法导入ABC Atlas Access，将使用本地数据模式")
                # 尝试导入pynwb
                try:
                    import pynwb
                    logger.info("已导入pynwb，可用于加载NWB文件数据")
                except ImportError:
                    logger.warning("无法导入pynwb，将使用纯本地CSV/Excel文件")

        # 层边界定义（相对深度）
        self.layer_boundaries = {
            'L1': (0.0, 0.1),
            'L2/3': (0.1, 0.35),
            'L4': (0.35, 0.5),
            'L5': (0.5, 0.7),
            'L6': (0.7, 0.9),
            'L6b': (0.9, 1.0)
        }

        # 层别名映射
        self.layer_alias = {
            'L2': 'L2/3',
            'L3': 'L2/3',
            'L23': 'L2/3',
            'L2-3': 'L2/3',
            'L6a': 'L6',
            'L6B': 'L6b',
            'Layer 1': 'L1',
            'Layer 2/3': 'L2/3',
            'Layer 4': 'L4',
            'Layer 5': 'L5',
            'Layer 6': 'L6',
            'Layer 6b': 'L6b'
        }

        # 投射类型标记基因映射 - 扩展了标记列表
        self.projection_markers = {
            'IT': ['Satb2', 'Cux1', 'Cux2', 'Plxnd1', 'Rorb', 'Lhx2', 'Rasgrf2', 'Slc30a3'],
            'ET': ['Fezf2', 'Bcl11b', 'Crym', 'Foxo1', 'Tbr1', 'Epha4', 'Tle4', 'Pcp4'],
            'CT': ['Tle4', 'Foxp2', 'Syt6', 'Ntsr1', 'Crym', 'Grik1', 'Tshz2'],
            'NP': ['Npr3', 'Cplx3', 'Rspo1', 'Rxfp1', 'Penk']  # Near-projecting
        }

        # 投射类型正则表达式模式（用于名称匹配）
        self.projection_regex = {
            'IT': re.compile(r'(IT|ipsilateral|intratelencephalic)', re.IGNORECASE),
            'ET': re.compile(r'(ET|PT|extratelencephalic|pyramidal|contralateral)', re.IGNORECASE),
            'CT': re.compile(r'(CT|corticothalamic|thalamic)', re.IGNORECASE),
            'NP': re.compile(r'(NP|near[- ]projecting)', re.IGNORECASE)
        }

    def _create_sample_data(self, region_acronym: str) -> None:
        """Removed sample data creation - now raises error when real data is missing"""
        raise FileNotFoundError(f"无法获取{region_acronym}的真实MERFISH数据。请确保数据文件存在并路径正确。")

    def load_merfish_data(self, region_acronym: str) -> pd.DataFrame:
        """加载特定脑区的MERFISH数据"""
        cache_file = self.cache_dir / f"merfish_{region_acronym}{self.job_id}.pkl"

        if cache_file.exists():
            logger.info(f"从缓存加载{region_acronym}的MERFISH数据...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info(f"尝试获取{region_acronym}的MERFISH数据...")

        cell_df = None

        # 首先尝试从本地cell_metadata文件加载
        try:
            metadata_files = []
            for i in range(1, 5):
                filepath = f"cell_metadata_with_cluster_annotation_{i}.csv"
                if os.path.exists(filepath):
                    metadata_files.append(filepath)

            if metadata_files:
                # 读取并合并所有元数据文件
                dfs = []
                for file in metadata_files:
                    logger.info(f"从本地文件{file}加载数据")
                    df = pd.read_csv(file)
                    # 检查是否有区域信息列
                    if 'region' in df.columns:
                        dfs.append(df)
                    elif 'Region' in df.columns:
                        df = df.rename(columns={'Region': 'region'})
                        dfs.append(df)
                    elif 'region_acronym' in df.columns:
                        df = df.rename(columns={'region_acronym': 'region'})
                        dfs.append(df)
                    else:
                        logger.warning(f"文件{file}中未找到区域信息列")

                if dfs:
                    # 合并所有DataFrame
                    cell_df = pd.concat(dfs, ignore_index=True)
                    # 筛选指定区域
                    cell_df = cell_df[cell_df['region'].str.contains(region_acronym, na=False)]
                    logger.info(f"从本地文件加载{region_acronym}数据，共{len(cell_df)}个细胞")
        except Exception as e:
            logger.warning(f"从本地文件加载{region_acronym}数据失败: {e}")

        # 尝试加载坐标数据并合并
        if cell_df is not None:
            try:
                coord_files = []
                for i in range(1, 5):
                    filepath = f"ccf_coordinates_{i}.csv"
                    if os.path.exists(filepath):
                        coord_files.append(filepath)

                if coord_files:
                    # 读取并合并所有坐标文件
                    coord_dfs = []
                    for file in coord_files:
                        logger.info(f"从本地文件{file}加载坐标数据")
                        coord_df = pd.read_csv(file)
                        coord_dfs.append(coord_df)

                    if coord_dfs:
                        # 合并所有坐标DataFrame
                        all_coords = pd.concat(coord_dfs, ignore_index=True)

                        # 确保有细胞ID列用于合并
                        if 'cell_id' in all_coords.columns and 'cell_id' in cell_df.columns:
                            cell_df = pd.merge(cell_df, all_coords, on='cell_id', how='left')
                            logger.info(f"合并了{len(all_coords)}条坐标数据")
                        elif 'cell_id' in all_coords.columns:
                            # 尝试其他可能的ID列名
                            for id_col in ['id', 'ID', 'CellID', 'cell_ID']:
                                if id_col in cell_df.columns:
                                    cell_df = cell_df.rename(columns={id_col: 'cell_id'})
                                    cell_df = pd.merge(cell_df, all_coords, on='cell_id', how='left')
                                    logger.info(f"使用{id_col}列合并了坐标数据")
                                    break
            except Exception as e:
                logger.warning(f"加载坐标数据失败: {e}")

        # 尝试加载h5ad文件中的基因表达数据
        if cell_df is not None:
            try:
                h5ad_files = []
                for i in range(1, 5):
                    log2_file = f"Zhuang-ABCA-{i}-log2.h5ad"
                    if os.path.exists(log2_file):
                        h5ad_files.append(log2_file)

                if h5ad_files:
                    try:
                        import scanpy as sc
                        import anndata

                        # 读取h5ad文件并提取基因表达数据
                        for file in h5ad_files:
                            logger.info(f"从{file}加载基因表达数据")
                            adata = sc.read_h5ad(file)

                            # 尝试匹配细胞ID
                            common_cells = set(adata.obs.index).intersection(set(cell_df['cell_id']))
                            if common_cells:
                                # 提取共同细胞的基因表达
                                adata = adata[list(common_cells)]
                                expr_df = pd.DataFrame(adata.X.toarray(), index=adata.obs.index,
                                                       columns=adata.var.index)

                                # 将基因表达数据合并到cell_df
                                for gene in adata.var.index:
                                    cell_df.loc[cell_df['cell_id'].isin(common_cells), gene] = expr_df[gene]

                                logger.info(f"合并了{len(common_cells)}个细胞的基因表达数据")
                    except ImportError:
                        logger.warning("未安装scanpy或anndata，无法读取h5ad文件")
                    except Exception as e:
                        logger.warning(f"读取h5ad文件失败: {e}")
            except Exception as e:
                logger.warning(f"加载基因表达数据失败: {e}")

        # 如果未获取到数据，报错
        if cell_df is None or len(cell_df) == 0:
            raise FileNotFoundError(f"无法获取{region_acronym}的真实MERFISH数据。请确保数据文件存在并路径正确。")

        # 确保有基本列
        for col in ['layer', 'class', 'subclass', 'cluster']:
            if col not in cell_df.columns:
                cell_df[col] = np.nan

        # 保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(cell_df, f)

        return cell_df

    def assign_layer_to_cells(self, cell_df: pd.DataFrame, region_name: str) -> pd.DataFrame:
        """为细胞分配层信息"""
        # 如果已有层信息，统一层名称格式
        if 'layer' in cell_df.columns:
            # 应用层别名映射
            cell_df['layer'] = cell_df['layer'].apply(
                lambda x: self.layer_alias.get(str(x), str(x)) if pd.notna(x) else np.nan
            )

            # 计算已有层标签的比例
            layer_count = cell_df['layer'].notna().sum()
            logger.info(f"{region_name}数据中已有{layer_count}/{len(cell_df)}个细胞的层标签")

            # 如果超过50%的细胞已有层标签，使用现有标签
            if layer_count / len(cell_df) > 0.5:
                # 填充缺失的层标签
                missing_layers = cell_df['layer'].isna()
                if missing_layers.any():
                    logger.info(f"填充{missing_layers.sum()}个缺失的层标签")
                    self._assign_missing_layers(cell_df)
                return cell_df

        # 计算相对深度
        if 'depth' in cell_df.columns and cell_df['depth'].notna().any():
            # 使用深度数据
            depth_col = 'depth'
            max_depth = cell_df[depth_col].max()
            min_depth = cell_df[depth_col].min()
            if max_depth > min_depth:
                cell_df['relative_depth'] = (cell_df[depth_col] - min_depth) / (max_depth - min_depth)
            else:
                cell_df['relative_depth'] = 0.5
        elif 'y' in cell_df.columns and cell_df['y'].notna().any():
            # 使用y坐标（假设y坐标与深度相关）
            depth_col = 'y'
            max_y = cell_df[depth_col].max()
            min_y = cell_df[depth_col].min()
            if max_y > min_y:
                # 通常y坐标向下增加，所以用1减去归一化值
                cell_df['relative_depth'] = 1 - (cell_df[depth_col] - min_y) / (max_y - min_y)
            else:
                cell_df['relative_depth'] = 0.5
        else:
            # 无法确定深度，使用默认值
            logger.warning(f"{region_name}数据缺少深度信息，使用默认层分配")
            cell_df['relative_depth'] = 0.5

        # 分配层
        cell_df['layer'] = cell_df['relative_depth'].apply(self._depth_to_layer)

        return cell_df

    def _depth_to_layer(self, depth):
        """根据相对深度分配层"""
        if pd.isna(depth):
            return 'L5'  # 默认层

        for layer, (min_d, max_d) in self.layer_boundaries.items():
            if min_d <= depth < max_d:
                return layer
        return 'L5'  # 默认层

    def _assign_missing_layers(self, cell_df: pd.DataFrame):
        """填充缺失的层标签"""
        # 计算每个细胞类型最常见的层
        if 'subclass' in cell_df.columns and cell_df['subclass'].notna().any():
            # 按subclass分组
            subclass_layer = cell_df[cell_df['layer'].notna()].groupby('subclass')['layer'].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 'L5'
            ).to_dict()

            # 填充缺失值
            for subclass, layer in subclass_layer.items():
                mask = (cell_df['layer'].isna()) & (cell_df['subclass'] == subclass)
                cell_df.loc[mask, 'layer'] = layer

        # 填充剩余缺失值
        cell_df['layer'] = cell_df['layer'].fillna('L5')

    def determine_projection_type(self, df_row) -> str:
        """根据标记基因表达和名称确定投射类型"""
        # 1. 首先尝试从亚类名称匹配
        if 'subclass' in df_row and pd.notna(df_row['subclass']):
            subclass_name = str(df_row['subclass'])

            for proj_type, pattern in self.projection_regex.items():
                if pattern.search(subclass_name):
                    return proj_type

        # 2. 然后尝试使用标记基因表达
        markers_found = defaultdict(int)

        for proj_type, marker_genes in self.projection_markers.items():
            for gene in marker_genes:
                if gene in df_row and pd.notna(df_row[gene]) and df_row[gene] > 0:
                    markers_found[proj_type] += 1

        if markers_found:
            # 返回得分最高的类型
            return max(markers_found.items(), key=lambda x: x[1])[0]

        # 3. 使用层位置启发式规则作为后备
        if 'layer' in df_row and pd.notna(df_row['layer']):
            layer = df_row['layer']
            if layer in ['L2/3', 'L4', 'L6']:
                return 'IT'  # 大多数L2/3、L4和L6是IT神经元
            elif layer == 'L5':
                return 'ET'  # L5包含许多ET神经元
            elif layer == 'L6b':
                return 'CT'  # L6b包含许多CT神经元

        return 'UNK'

    def calculate_gene_coexpression(self, cell_df: pd.DataFrame,
                                    core_genes: List[str] = None) -> pd.DataFrame:
        """计算基因间共表达关系"""
        logger.info("计算基因共表达关系...")

        # 如果未指定核心基因，使用投射类型标记基因
        if core_genes is None:
            core_genes = []
            for genes in self.projection_markers.values():
                core_genes.extend(genes)
            # 添加一些重要的神经元类型标记基因
            core_genes.extend(['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Chodl', 'Calb2', 'Nos1'])
            # 添加Fezf2模块基因
            core_genes.extend(['Fezf2', 'Sox5', 'Zfpm2', 'Rxfp1', 'Ctgf'])
            # 去重
            core_genes = list(set(core_genes))

        # 筛选细胞数据中存在的基因列
        available_genes = [gene for gene in core_genes if gene in cell_df.columns]

        if len(available_genes) < 2:
            logger.warning(f"可用基因数量不足，无法计算共表达关系。找到{len(available_genes)}个基因。")
            return pd.DataFrame()

        logger.info(f"使用{len(available_genes)}个基因计算共表达关系")

        # 提取基因表达数据
        gene_expr = cell_df[available_genes]

        # 计算Spearman相关系数
        corr_matrix = gene_expr.corr(method='spearman')

        # 计算p值
        p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)

        for i, gene1 in enumerate(available_genes):
            for j, gene2 in enumerate(available_genes):
                if i >= j:  # 只计算矩阵的上三角部分
                    continue

                # 获取非零表达值
                mask = (gene_expr[gene1].notna()) & (gene_expr[gene2].notna())
                x = gene_expr.loc[mask, gene1]
                y = gene_expr.loc[mask, gene2]

                if len(x) > 5:  # 确保有足够的数据点
                    _, p_value = stats.spearmanr(x, y)
                    p_values.loc[gene1, gene2] = p_value
                    p_values.loc[gene2, gene1] = p_value
                else:
                    p_values.loc[gene1, gene2] = np.nan
                    p_values.loc[gene2, gene1] = np.nan

        # 计算FDR校正的p值
        flat_p_values = p_values.values[np.triu_indices_from(p_values.values, k=1)]
        flat_p_values = flat_p_values[~np.isnan(flat_p_values)]

        if len(flat_p_values) > 0:
            # 执行FDR校正
            try:
                _, flat_fdr = stats.false_discovery_control(flat_p_values)

                # 重建FDR矩阵
                fdr_values = pd.DataFrame(np.nan, index=p_values.index, columns=p_values.columns)
                idx = 0
                for i in range(len(available_genes)):
                    for j in range(i + 1, len(available_genes)):
                        if not np.isnan(p_values.iloc[i, j]):
                            fdr_values.iloc[i, j] = flat_fdr[idx]
                            fdr_values.iloc[j, i] = flat_fdr[idx]
                            idx += 1
            except:
                # 如果stats.false_discovery_control不可用，使用简单的Bonferroni校正
                fdr_values = p_values.copy() * len(flat_p_values)
                fdr_values = fdr_values.clip(upper=1.0)
        else:
            fdr_values = pd.DataFrame(np.nan, index=p_values.index, columns=p_values.columns)

        # 转换为边列表格式
        coexpr_edges = []

        for i, gene1 in enumerate(available_genes):
            for j, gene2 in enumerate(available_genes):
                if i >= j:  # 只使用矩阵的上三角部分
                    continue

                rho = corr_matrix.loc[gene1, gene2]
                p_val = p_values.loc[gene1, gene2]
                fdr = fdr_values.loc[gene1, gene2]

                if pd.notna(rho) and pd.notna(p_val) and pd.notna(fdr):
                    coexpr_edges.append({
                        'gene1': gene1,
                        'gene2': gene2,
                        'rho': rho,
                        'p_value': p_val,
                        'fdr': fdr
                    })

        # 转换为DataFrame
        result_df = pd.DataFrame(coexpr_edges)

        # 按相关系数绝对值排序
        if not result_df.empty:
            result_df['abs_rho'] = result_df['rho'].abs()
            result_df = result_df.sort_values('abs_rho', ascending=False)
            result_df = result_df.drop('abs_rho', axis=1)

        return result_df

    def calculate_gene_expression_by_regionlayer(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        """计算每个RegionLayer中基因的平均表达量"""
        logger.info("计算RegionLayer基因表达...")

        # 首先确定哪些列是基因
        all_columns = set(cell_df.columns)
        non_gene_columns = {'cell_id', 'region', 'layer', 'x', 'y', 'z', 'depth',
                            'class', 'subclass', 'cluster', 'relative_depth',
                            'proj_type', 'depth_position'}

        gene_columns = list(all_columns - non_gene_columns)

        # 检查是否有基因列
        if not gene_columns:
            logger.warning("未找到基因表达列，无法计算RegionLayer基因表达")
            return pd.DataFrame()

        # 对数据进行分组并计算平均表达量
        logger.info(f"计算{len(gene_columns)}个基因在各RegionLayer的平均表达量")

        # 按region和layer分组
        grouped = cell_df.groupby(['region', 'layer'])

        # 初始化结果列表
        results = []

        # 为每个region-layer计算基因表达
        for (region, layer), group in grouped:
            rl_id = f"{region}_{layer}"

            # 计算每个基因的平均表达量（使用logCPM值或原始值）
            for gene in gene_columns:
                expr_values = group[gene].dropna()

                if len(expr_values) > 0:
                    # 计算平均值
                    mean_expr = expr_values.mean()

                    # 只保留有表达的基因
                    if mean_expr > 0:
                        results.append({
                            'rl_id': rl_id,
                            'region': region,
                            'layer': layer,
                            'gene': gene,
                            'mean_logCPM': mean_expr,
                            'n_cells': len(expr_values)
                        })

        return pd.DataFrame(results)

    def calculate_gene_coexpression(self, cell_df: pd.DataFrame,
                                    core_genes: List[str] = None) -> pd.DataFrame:
        """计算基因间共表达关系"""
        logger.info("计算基因共表达关系...")

        # 如果未指定核心基因，使用投射类型标记基因
        if core_genes is None:
            core_genes = []
            for genes in self.projection_markers.values():
                core_genes.extend(genes)
            # 添加一些重要的神经元类型标记基因
            core_genes.extend(['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Chodl', 'Calb2', 'Nos1'])
            # 添加Fezf2模块基因
            core_genes.extend(['Fezf2', 'Sox5', 'Zfpm2', 'Rxfp1', 'Ctgf'])
            # 去重
            core_genes = list(set(core_genes))

        # 筛选细胞数据中存在的基因列
        available_genes = [gene for gene in core_genes if gene in cell_df.columns]

        if len(available_genes) < 2:
            logger.warning(f"可用基因数量不足，无法计算共表达关系。找到{len(available_genes)}个基因。")
            return pd.DataFrame()

        logger.info(f"使用{len(available_genes)}个基因计算共表达关系")

        # 提取基因表达数据
        gene_expr = cell_df[available_genes]

        # 计算Spearman相关系数
        corr_matrix = gene_expr.corr(method='spearman')

        # 计算p值
        p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)

        for i, gene1 in enumerate(available_genes):
            for j, gene2 in enumerate(available_genes):
                if i >= j:  # 只计算矩阵的上三角部分
                    continue

                # 获取非零表达值
                mask = (gene_expr[gene1].notna()) & (gene_expr[gene2].notna())
                x = gene_expr.loc[mask, gene1]
                y = gene_expr.loc[mask, gene2]

                if len(x) > 5:  # 确保有足够的数据点
                    rho, p_value = stats.spearmanr(x, y)
                    p_values.loc[gene1, gene2] = p_value
                    p_values.loc[gene2, gene1] = p_value
                else:
                    p_values.loc[gene1, gene2] = np.nan
                    p_values.loc[gene2, gene1] = np.nan

        # 计算FDR校正的p值
        flat_p_values = p_values.values[np.triu_indices_from(p_values.values, k=1)]
        flat_p_values = flat_p_values[~np.isnan(flat_p_values)]

        if len(flat_p_values) > 0:
            # 执行FDR校正
            try:
                _, flat_fdr = stats.false_discovery_control(flat_p_values)

                # 重建FDR矩阵
                fdr_values = pd.DataFrame(np.nan, index=p_values.index, columns=p_values.columns)
                idx = 0
                for i in range(len(available_genes)):
                    for j in range(i + 1, len(available_genes)):
                        if not np.isnan(p_values.iloc[i, j]):
                            fdr_values.iloc[i, j] = flat_fdr[idx]
                            fdr_values.iloc[j, i] = flat_fdr[idx]
                            idx += 1
            except:
                # 如果stats.false_discovery_control不可用，使用Benjamini-Hochberg校正
                sorted_idx = np.argsort(flat_p_values)
                sorted_p = flat_p_values[sorted_idx]

                # Benjamini-Hochberg校正
                n = len(sorted_p)
                bh_thresholds = np.arange(1, n + 1) / n * 0.05
                significant = sorted_p <= bh_thresholds

                # 找到最大显著索引
                max_idx = np.max(np.where(significant)[0]) if any(significant) else -1

                # 计算校正后的p值
                flat_fdr = np.ones_like(flat_p_values)
                flat_fdr[sorted_idx[:max_idx + 1]] = sorted_p[:max_idx + 1]

                # 重建FDR矩阵
                fdr_values = pd.DataFrame(np.nan, index=p_values.index, columns=p_values.columns)
                idx = 0
                for i in range(len(available_genes)):
                    for j in range(i + 1, len(available_genes)):
                        if not np.isnan(p_values.iloc[i, j]):
                            fdr_values.iloc[i, j] = flat_fdr[idx]
                            fdr_values.iloc[j, i] = flat_fdr[idx]
                            idx += 1
        else:
            fdr_values = pd.DataFrame(np.nan, index=p_values.index, columns=p_values.columns)

        # 转换为边列表格式
        coexpr_edges = []

        for i, gene1 in enumerate(available_genes):
            for j, gene2 in enumerate(available_genes):
                if i >= j:  # 只使用矩阵的上三角部分
                    continue

                rho = corr_matrix.loc[gene1, gene2]
                p_val = p_values.loc[gene1, gene2]
                fdr = fdr_values.loc[gene1, gene2]

                if pd.notna(rho) and pd.notna(p_val) and pd.notna(fdr):
                    coexpr_edges.append({
                        'gene1': gene1,
                        'gene2': gene2,
                        'rho': float(rho),  # 确保是Python原生类型
                        'p_value': float(p_val),
                        'fdr': float(fdr)
                    })

        # 转换为DataFrame
        result_df = pd.DataFrame(coexpr_edges)

        # 按相关系数绝对值排序
        if not result_df.empty:
            result_df['abs_rho'] = result_df['rho'].abs()
            result_df = result_df.sort_values('abs_rho', ascending=False)
            result_df = result_df.drop('abs_rho', axis=1)

        return result_df

    def export_transcriptomic_relationships(self, distributions: Dict[str, Dict[str, Any]],
                                            region_id_map: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
        """导出转录组关系数据为CSV文件"""
        logger.info("导出转录组关系数据...")

        # 创建各关系类型的空DataFrame
        relationship_dfs = {
            'has_class': pd.DataFrame(columns=['rl_id', 'class_name', 'pct_cells', 'rank', 'n_cells']),
            'has_subclass': pd.DataFrame(
                columns=['rl_id', 'subclass_name', 'pct_cells', 'rank', 'n_cells', 'proj_type']),
            'has_cluster': pd.DataFrame(columns=['rl_id', 'cluster_name', 'pct_cells', 'rank', 'n_cells'])
        }

        # RegionLayer节点属性
        region_layer_df = pd.DataFrame(columns=['rl_id', 'region_name', 'layer', 'region_id',
                                                'it_pct', 'et_pct', 'ct_pct', 'lr_pct', 'lr_prior', 'total_cells'])

        # 导出亚类-投射类型映射
        subclass_projtype_df = pd.DataFrame(columns=['subclass_name', 'proj_type'])
        subclass_projtype_map = {}

        # 将分布数据转换为DataFrame
        for rl_id, dist_data in distributions.items():
            # RegionLayer属性
            region_name = dist_data.get('region_name', '')
            layer = dist_data.get('layer', '')
            region_id = region_id_map.get(region_name, '') if region_id_map else ''

            region_layer_df = pd.concat([region_layer_df, pd.DataFrame([{
                'rl_id': rl_id,
                'region_name': region_name,
                'layer': layer,
                'region_id': region_id,
                'it_pct': dist_data.get('it_pct', 0.0),
                'et_pct': dist_data.get('et_pct', 0.0),
                'ct_pct': dist_data.get('ct_pct', 0.0),
                'lr_pct': dist_data.get('lr_pct', 0.0),
                'lr_prior': dist_data.get('lr_prior', 0.0),
                'total_cells': dist_data.get('total_cells', 0)
            }])], ignore_index=True)

            # 处理Class关系
            if 'classes' in dist_data:
                for class_info in dist_data['classes']:
                    relationship_dfs['has_class'] = pd.concat([relationship_dfs['has_class'], pd.DataFrame([{
                        'rl_id': rl_id,
                        'class_name': class_info['class_name'],
                        'pct_cells': class_info['pct_cells'],
                        'rank': class_info['rank'],
                        'n_cells': class_info['count']
                    }])], ignore_index=True)

            # 处理Subclass关系
            if 'subclasses' in dist_data:
                for subclass_info in dist_data['subclasses']:
                    relationship_dfs['has_subclass'] = pd.concat([relationship_dfs['has_subclass'], pd.DataFrame([{
                        'rl_id': rl_id,
                        'subclass_name': subclass_info['subclass_name'],
                        'pct_cells': subclass_info['pct_cells'],
                        'rank': subclass_info['rank'],
                        'n_cells': subclass_info['count'],
                        'proj_type': subclass_info['proj_type']
                    }])], ignore_index=True)

                    # 更新亚类-投射类型映射
                    subclass_name = subclass_info['subclass_name']
                    proj_type = subclass_info['proj_type']

                    if subclass_name not in subclass_projtype_map:
                        subclass_projtype_map[subclass_name] = proj_type
                        subclass_projtype_df = pd.concat([subclass_projtype_df, pd.DataFrame([{
                            'subclass_name': subclass_name,
                            'proj_type': proj_type
                        }])], ignore_index=True)

            # 处理Cluster关系
            if 'clusters' in dist_data:
                for cluster_info in dist_data['clusters']:
                    relationship_dfs['has_cluster'] = pd.concat([relationship_dfs['has_cluster'], pd.DataFrame([{
                        'rl_id': rl_id,
                        'cluster_name': cluster_info['cluster_name'],
                        'pct_cells': cluster_info['pct_cells'],
                        'rank': cluster_info['rank'],
                        'n_cells': cluster_info['count']
                    }])], ignore_index=True)

        # 保存到CSV文件
        output_files = {}

        for rel_type, df in relationship_dfs.items():
            if not df.empty:
                output_file = self.output_dir / f"{rel_type}.csv"
                df.to_csv(output_file, index=False)
                output_files[rel_type] = output_file
                logger.info(f"已导出{len(df)}条{rel_type}关系到{output_file}")

        # 保存RegionLayer属性
        if not region_layer_df.empty:
            rl_output_file = self.output_dir / "region_layer_props.csv"
            region_layer_df.to_csv(rl_output_file, index=False)
            output_files['region_layer_props'] = rl_output_file
            logger.info(f"已导出{len(region_layer_df)}个RegionLayer属性到{rl_output_file}")

        # 保存亚类-投射类型映射
        if not subclass_projtype_df.empty:
            proj_type_file = self.output_dir / "subclass_projtype.csv"
            subclass_projtype_df.to_csv(proj_type_file, index=False)
            output_files['subclass_projtype'] = proj_type_file
            logger.info(f"已导出{len(subclass_projtype_df)}个亚类-投射类型映射到{proj_type_file}")

        # 导出基因共表达关系
        # 从每个区域收集细胞数据
        all_cells = []
        for region in set(region_layer_df['region_name']):
            try:
                cells = self.load_merfish_data(region)
                all_cells.append(cells)
                logger.info(f"从{region}加载了{len(cells)}个细胞用于计算基因共表达")
            except Exception as e:
                logger.warning(f"无法加载{region}的细胞数据: {e}")

        if all_cells:
            # 合并所有细胞数据
            all_cells_df = pd.concat(all_cells, ignore_index=True)
            logger.info(f"共合并{len(all_cells_df)}个细胞数据计算基因共表达")

            # 计算共表达关系
            coexpr_df = self.calculate_gene_coexpression(all_cells_df)

            if not coexpr_df.empty:
                # 保存共表达关系
                coexpr_file = self.output_dir / "gene_coexpression.csv"
                coexpr_df.to_csv(coexpr_file, index=False)
                output_files['gene_coexpression'] = coexpr_file
                logger.info(f"已导出{len(coexpr_df)}条基因共表达关系到{coexpr_file}")

        return output_files
    def calculate_cell_type_distribution(self, region_name: str) -> Dict[str, Dict[str, float]]:
        """计算特定脑区各层的细胞类型分布"""
        # 加载MERFISH数据
        try:
            cell_df = self.load_merfish_data(region_name)
        except Exception as e:
            logger.warning(f"无法加载{region_name}的MERFISH数据: {e}")
            return {}

        # 分配层信息
        cell_df = self.assign_layer_to_cells(cell_df, region_name)

        # 添加投射类型
        cell_df['proj_type'] = cell_df.apply(self.determine_projection_type, axis=1)

        # 计算基因共表达关系
        try:
            gene_coexpr_df = self.calculate_gene_coexpression(cell_df)
            if not gene_coexpr_df.empty:
                coexpr_file = self.output_dir / f"{region_name}_gene_coexpression.csv"
                gene_coexpr_df.to_csv(coexpr_file, index=False)
                logger.info(f"保存{len(gene_coexpr_df)}条基因共表达关系到{coexpr_file}")
        except Exception as e:
            logger.error(f"计算{region_name}的基因共表达关系失败: {e}")

        # 计算RegionLayer基因表达
        try:
            gene_expr_df = self.calculate_gene_expression_by_regionlayer(cell_df)
            if not gene_expr_df.empty:
                expr_file = self.output_dir / f"{region_name}_gene_expression.csv"
                gene_expr_df.to_csv(expr_file, index=False)
                logger.info(f"保存{len(gene_expr_df)}条RegionLayer基因表达数据到{expr_file}")
        except Exception as e:
            logger.error(f"计算{region_name}的RegionLayer基因表达失败: {e}")

        # 计算每层的细胞类型分布
        distributions = {}

        # 优化：一次性计算所有层的分布
        layer_groups = cell_df.groupby('layer')

        for layer, layer_cells in layer_groups:
            # 跳过不在预定义层中的数据
            if layer not in self.layer_boundaries and layer not in self.layer_alias.values():
                continue

            if len(layer_cells) == 0:
                continue

            # 标准化层名
            std_layer = self.layer_alias.get(layer, layer)
            rl_id = f"{region_name}_{std_layer}"

            # 统计各细胞类型
            distributions[rl_id] = {
                'total_cells': len(layer_cells),
                'region_name': region_name,
                'layer': std_layer
            }

            # 计算投射类型比例
            proj_counts = layer_cells['proj_type'].value_counts(normalize=True)
            distributions[rl_id]['it_pct'] = min(1.0, float(proj_counts.get('IT', 0)))
            distributions[rl_id]['et_pct'] = min(1.0, float(proj_counts.get('ET', 0)))
            distributions[rl_id]['ct_pct'] = min(1.0, float(proj_counts.get('CT', 0)))

            # 计算长程投射比例（使用ET作为代理）
            distributions[rl_id]['lr_pct'] = distributions[rl_id]['et_pct']

            # 计算长程投射优先级
            distributions[rl_id]['lr_prior'] = self._calculate_lr_priority(layer_cells)

            # 按class统计
            if 'class' in layer_cells.columns and layer_cells['class'].notna().any():
                class_counts = layer_cells['class'].value_counts(normalize=True)
                class_list = []

                for class_name, count in class_counts.items():
                    if pd.notna(class_name) and class_name:
                        class_info = {
                            'class_name': class_name,
                            'pct_cells': min(1.0, float(count)),
                            'count': int(len(layer_cells) * count)
                        }
                        class_list.append(class_info)

                # 排序并添加rank
                class_list.sort(key=lambda x: x['pct_cells'], reverse=True)
                for rank, class_info in enumerate(class_list, 1):
                    class_info['rank'] = rank

                distributions[rl_id]['classes'] = class_list

            # 按subclass统计
            if 'subclass' in layer_cells.columns and layer_cells['subclass'].notna().any():
                subclass_counts = layer_cells['subclass'].value_counts(normalize=True)
                subclass_list = []

                for subclass_name, count in subclass_counts.items():
                    if pd.notna(subclass_name) and subclass_name:
                        # 获取该亚类中最常见的投射类型
                        subclass_cells = layer_cells[layer_cells['subclass'] == subclass_name]
                        proj_type = subclass_cells['proj_type'].value_counts().index[0] if len(
                            subclass_cells) > 0 else 'UNK'

                        subclass_info = {
                            'subclass_name': subclass_name,
                            'pct_cells': min(1.0, float(count)),
                            'count': int(len(layer_cells) * count),
                            'proj_type': proj_type
                        }
                        subclass_list.append(subclass_info)

                # 排序并添加rank
                subclass_list.sort(key=lambda x: x['pct_cells'], reverse=True)
                for rank, subclass_info in enumerate(subclass_list, 1):
                    subclass_info['rank'] = rank

                distributions[rl_id]['subclasses'] = subclass_list

            # 按cluster统计
            if 'cluster' in layer_cells.columns and layer_cells['cluster'].notna().any():
                cluster_counts = layer_cells['cluster'].value_counts(normalize=True)
                cluster_list = []

                # 只保留前20个最丰富的cluster
                for cluster_name, count in cluster_counts.head(20).items():
                    if pd.notna(cluster_name) and cluster_name:
                        cluster_info = {
                            'cluster_name': cluster_name,
                            'pct_cells': min(1.0, float(count)),
                            'count': int(len(layer_cells) * count)
                        }
                        cluster_list.append(cluster_info)

                # 排序并添加rank
                cluster_list.sort(key=lambda x: x['pct_cells'], reverse=True)
                for rank, cluster_info in enumerate(cluster_list, 1):
                    cluster_info['rank'] = rank

                distributions[rl_id]['clusters'] = cluster_list

        return distributions

    def _calculate_lr_priority(self, cell_df: pd.DataFrame) -> float:
        """计算长程投射优先级，基于多种长程投射标记物"""
        # 长程投射相关标记
        lr_markers = ['Chodl', 'Nos1', 'Chrna2', 'Calca', 'Tac2']

        # 计算表达这些标记的细胞比例
        lr_scores = []
        for marker in lr_markers:
            if marker in cell_df.columns:
                expr_ratio = (cell_df[marker] > 0).mean()
                lr_scores.append(expr_ratio)

        # ET神经元比例也作为指标
        et_ratio = (cell_df['proj_type'] == 'ET').mean() if 'proj_type' in cell_df.columns else 0

        # 综合计算优先级
        priority = 0.0
        if lr_scores:
            priority = np.mean(lr_scores) * 0.5 + et_ratio * 0.5
        else:
            priority = et_ratio

        return min(1.0, float(priority))

    def export_transcriptomic_relationships(self, distributions: Dict[str, Dict[str, Any]],
                                            region_id_map: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
        """导出转录组关系数据为CSV文件"""
        logger.info("导出转录组关系数据...")

        # 创建各关系类型的空DataFrame
        relationship_dfs = {
            'has_class': pd.DataFrame(columns=['rl_id', 'class_name', 'pct_cells', 'rank', 'n_cells']),
            'has_subclass': pd.DataFrame(
                columns=['rl_id', 'subclass_name', 'pct_cells', 'rank', 'n_cells', 'proj_type']),
            'has_cluster': pd.DataFrame(columns=['rl_id', 'cluster_name', 'pct_cells', 'rank', 'n_cells'])
        }

        # RegionLayer节点属性
        region_layer_df = pd.DataFrame(columns=['rl_id', 'region_name', 'layer', 'region_id',
                                                'it_pct', 'et_pct', 'ct_pct', 'lr_pct', 'lr_prior', 'total_cells'])

        # 导出亚类-投射类型映射
        subclass_projtype_df = pd.DataFrame(columns=['subclass_name', 'proj_type'])
        subclass_projtype_map = {}

        # 将分布数据转换为DataFrame
        for rl_id, dist_data in distributions.items():
            # RegionLayer属性
            region_name = dist_data.get('region_name', '')
            layer = dist_data.get('layer', '')
            region_id = region_id_map.get(region_name, '') if region_id_map else ''

            region_layer_df = pd.concat([region_layer_df, pd.DataFrame([{
                'rl_id': rl_id,
                'region_name': region_name,
                'layer': layer,
                'region_id': region_id,
                'it_pct': dist_data.get('it_pct', 0.0),
                'et_pct': dist_data.get('et_pct', 0.0),
                'ct_pct': dist_data.get('ct_pct', 0.0),
                'lr_pct': dist_data.get('lr_pct', 0.0),
                'lr_prior': dist_data.get('lr_prior', 0.0),
                'total_cells': dist_data.get('total_cells', 0)
            }])], ignore_index=True)

            # 处理Class关系
            if 'classes' in dist_data:
                for class_info in dist_data['classes']:
                    relationship_dfs['has_class'] = pd.concat([relationship_dfs['has_class'], pd.DataFrame([{
                        'rl_id': rl_id,
                        'class_name': class_info['class_name'],
                        'pct_cells': class_info['pct_cells'],
                        'rank': class_info['rank'],
                        'n_cells': class_info['count']
                    }])], ignore_index=True)

            # 处理Subclass关系
            if 'subclasses' in dist_data:
                for subclass_info in dist_data['subclasses']:
                    relationship_dfs['has_subclass'] = pd.concat([relationship_dfs['has_subclass'], pd.DataFrame([{
                        'rl_id': rl_id,
                        'subclass_name': subclass_info['subclass_name'],
                        'pct_cells': subclass_info['pct_cells'],
                        'rank': subclass_info['rank'],
                        'n_cells': subclass_info['count'],
                        'proj_type': subclass_info['proj_type']
                    }])], ignore_index=True)

                    # 更新亚类-投射类型映射
                    subclass_name = subclass_info['subclass_name']
                    proj_type = subclass_info['proj_type']

                    if subclass_name not in subclass_projtype_map:
                        subclass_projtype_map[subclass_name] = proj_type
                        subclass_projtype_df = pd.concat([subclass_projtype_df, pd.DataFrame([{
                            'subclass_name': subclass_name,
                            'proj_type': proj_type
                        }])], ignore_index=True)

            # 处理Cluster关系
            if 'clusters' in dist_data:
                for cluster_info in dist_data['clusters']:
                    relationship_dfs['has_cluster'] = pd.concat([relationship_dfs['has_cluster'], pd.DataFrame([{
                        'rl_id': rl_id,
                        'cluster_name': cluster_info['cluster_name'],
                        'pct_cells': cluster_info['pct_cells'],
                        'rank': cluster_info['rank'],
                        'n_cells': cluster_info['count']
                    }])], ignore_index=True)

        # 保存到CSV文件
        output_files = {}

        for rel_type, df in relationship_dfs.items():
            if not df.empty:
                output_file = self.output_dir / f"{rel_type}.csv"
                df.to_csv(output_file, index=False)
                output_files[rel_type] = output_file
                logger.info(f"已导出{len(df)}条{rel_type}关系到{output_file}")

        # 保存RegionLayer属性
        if not region_layer_df.empty:
            rl_output_file = self.output_dir / "region_layer_props.csv"
            region_layer_df.to_csv(rl_output_file, index=False)
            output_files['region_layer_props'] = rl_output_file
            logger.info(f"已导出{len(region_layer_df)}个RegionLayer属性到{rl_output_file}")

        # 保存亚类-投射类型映射
        if not subclass_projtype_df.empty:
            proj_type_file = self.output_dir / "subclass_projtype.csv"
            subclass_projtype_df.to_csv(proj_type_file, index=False)
            output_files['subclass_projtype'] = proj_type_file
            logger.info(f"已导出{len(subclass_projtype_df)}个亚类-投射类型映射到{proj_type_file}")

        # 汇总所有区域的基因共表达关系
        try:
            coexpr_files = list(self.output_dir.glob("*_gene_coexpression.csv"))
            if coexpr_files:
                all_coexpr_dfs = []
                for file in coexpr_files:
                    coexpr_df = pd.read_csv(file)
                    all_coexpr_dfs.append(coexpr_df)

                if all_coexpr_dfs:
                    combined_coexpr = pd.concat(all_coexpr_dfs, ignore_index=True)
                    # 按gene1, gene2分组并取平均值
                    grouped = combined_coexpr.groupby(['gene1', 'gene2']).agg({
                        'rho': 'mean',
                        'p_value': 'min',
                        'fdr': 'min'
                    }).reset_index()

                    # 保存汇总的共表达关系
                    coexpr_output = self.output_dir / "gene_coexpression.csv"
                    grouped.to_csv(coexpr_output, index=False)
                    output_files['gene_coexpression'] = coexpr_output
                    logger.info(f"已导出{len(grouped)}条合并的基因共表达关系到{coexpr_output}")
        except Exception as e:
            logger.error(f"汇总基因共表达关系失败: {e}")

        # 汇总所有区域的基因表达数据
        try:
            expr_files = list(self.output_dir.glob("*_gene_expression.csv"))
            if expr_files:
                all_expr_dfs = []
                for file in expr_files:
                    expr_df = pd.read_csv(file)
                    all_expr_dfs.append(expr_df)

                if all_expr_dfs:
                    combined_expr = pd.concat(all_expr_dfs, ignore_index=True)

                    # 保存汇总的基因表达数据
                    expr_output = self.output_dir / "gene_expression.csv"
                    combined_expr.to_csv(expr_output, index=False)
                    output_files['gene_expression'] = expr_output
                    logger.info(f"已导出{len(combined_expr)}条合并的基因表达数据到{expr_output}")
        except Exception as e:
            logger.error(f"汇总基因表达数据失败: {e}")

        return output_files

    def process_regions(self, regions: List[str], region_id_map: Dict[str, str] = None) -> Dict[str, Any]:
        """处理多个脑区，计算分布并导出结果"""
        start_time = time.time()
        logger.info(f"开始处理{len(regions)}个脑区...")

        all_distributions = {}

        for region in regions:
            logger.info(f"处理脑区: {region}")
            distributions = self.calculate_cell_type_distribution(region)
            all_distributions.update(distributions)

        # 导出结果
        output_files = self.export_transcriptomic_relationships(all_distributions, region_id_map)

        elapsed_time = time.time() - start_time
        logger.info(f"处理完成。耗时: {elapsed_time:.2f}秒")

        return {
            'distributions': all_distributions,
            'output_files': output_files,
            'elapsed_time': elapsed_time
        }


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='MERFISH数据集成工具')
    parser.add_argument('--regions', '-r', nargs='+', default=['MOp', 'SSp', 'VISp'],
                        help='要处理的脑区列表')
    parser.add_argument('--cache-dir', '-c', help='缓存目录路径，默认使用系统临时目录')
    parser.add_argument('--output-dir', '-o', default='merfish_output',
                        help='输出CSV文件的目录')
    parser.add_argument('--region-id-map', '-m', help='区域名称到ID的映射文件（CSV格式）')
    parser.add_argument('--no-job-id', action='store_true',
                        help='不在缓存文件名中包含作业ID')

    args = parser.parse_args()

    # 加载区域ID映射
    region_id_map = None
    if args.region_id_map and os.path.exists(args.region_id_map):
        try:
            region_map_df = pd.read_csv(args.region_id_map)
            if 'name' in region_map_df.columns and 'id' in region_map_df.columns:
                region_id_map = dict(zip(region_map_df['name'], region_map_df['id']))
                logger.info(f"已加载{len(region_id_map)}个区域ID映射")
        except Exception as e:
            logger.warning(f"加载区域ID映射失败: {e}")

    # 初始化MERFISH集成器
    integrator = MERFISHDataIntegration(
        cache_dir=args.cache_dir,
        include_job_id=not args.no_job_id,
        output_dir=args.output_dir
    )

    # 处理脑区
    results = integrator.process_regions(args.regions, region_id_map)

    # 打印摘要
    print("\n处理摘要:")
    print(f"处理了 {len(args.regions)} 个脑区")
    print(f"生成了 {len(results['distributions'])} 个RegionLayer分布")
    print(f"输出文件保存在: {args.output_dir}")
    print(f"总耗时: {results['elapsed_time']:.2f}秒")


if __name__ == "__main__":
    main()