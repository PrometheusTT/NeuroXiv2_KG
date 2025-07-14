import json
import nrrd
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

    def __init__(self, data_dir: str = ".", output_dir: str = "merfish_output", job_id: str = ""):
        """
        初始化MERFISH数据集成工具

        Args:
            data_dir: 数据目录，默认为当前目录
            output_dir: 输出目录，默认为 "merfish_output"
            job_id: 可选的作业ID后缀，用于区分不同运行的输出文件
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.job_id = job_id

        # 添加缺失的cache_dir属性
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 设置日志
        self.logger = logging.getLogger("merfish_integration")

        # 记录初始化信息
        self.logger.info(f"初始化MERFISH数据集成工具")
        self.logger.info(f"数据目录: {self.data_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"缓存目录: {self.cache_dir}")
        if job_id:
            self.logger.info(f"作业ID: {job_id}")

    def process_region(self, region):
        """处理单个脑区数据"""
        self.logger.info(f"处理脑区: {region}")

        # 创建区域输出目录
        region_dir = os.path.join(self.output_dir, region)
        os.makedirs(region_dir, exist_ok=True)

        try:
            # 1. 初始化区域数据结构
            region_data = {
                "region_name": region,
                "layers": ["L1", "L2/3", "L4", "L5", "L6", "L6b"],
                "cell_count": 0,
                "has_merfish_data": False
            }

            # 2. 创建区域层属性数据
            layers = region_data["layers"]
            rl_props_rows = []

            for layer in layers:
                # 为每个层创建默认属性
                rl_id = f"{region}_{layer}"

                # 不同层的投射类型分布有所不同
                if layer == "L2/3":
                    it_pct, et_pct, ct_pct = 0.7, 0.2, 0.1  # L2/3以IT为主
                elif layer == "L5":
                    it_pct, et_pct, ct_pct = 0.4, 0.5, 0.1  # L5以ET为主
                elif layer == "L6":
                    it_pct, et_pct, ct_pct = 0.3, 0.2, 0.5  # L6以CT为主
                else:
                    it_pct, et_pct, ct_pct = 0.5, 0.3, 0.2  # 其他层默认分布

                row = {
                    "rl_id": rl_id,
                    "region_name": region,
                    "layer": layer,
                    "n_neuron": 100,  # 默认神经元数量
                    "it_pct": it_pct,
                    "et_pct": et_pct,
                    "ct_pct": ct_pct,
                    "lr_pct": et_pct,  # 长程投射与ET相同
                    "lr_prior": 0.2,
                    "morph_ax_len_mean": 1000.0,  # 默认轴突长度
                    "morph_ax_len_std": 200.0,
                    "dend_polarity_index_mean": 0.7,  # 默认树突极性
                    "dend_br_std": 5.0  # 默认树突分支标准差
                }
                rl_props_rows.append(row)

            # 保存RegionLayer属性
            rl_props_file = os.path.join(region_dir, "region_layer_props.csv")
            pd.DataFrame(rl_props_rows).to_csv(rl_props_file, index=False)
            self.logger.info(f"保存区域层属性到: {rl_props_file}")

            # 3. 创建基本的转录组关系数据
            for rel_type in ['class', 'subclass', 'cluster']:
                rel_rows = []
                for layer in layers:
                    rl_id = f"{region}_{layer}"

                    # 不同层的细胞类型分布有所不同
                    if rel_type == 'class':
                        # 类别关系 - 主要是谷氨酸能和GABA能神经元
                        rel_rows.extend([
                            {"rl_id": rl_id, "class_name": "Glutamatergic", "pct_cells": 0.8, "rank": 1, "n_cells": 80},
                            {"rl_id": rl_id, "class_name": "GABAergic", "pct_cells": 0.2, "rank": 2, "n_cells": 20}
                        ])
                    elif rel_type == 'subclass':
                        # 亚类关系 - 包括不同投射类型
                        if layer == "L2/3":
                            rel_rows.extend([
                                {"rl_id": rl_id, "subclass_name": "L23_IT", "pct_cells": 0.6, "rank": 1, "n_cells": 60,
                                 "proj_type": "IT"},
                                {"rl_id": rl_id, "subclass_name": "L5_IT", "pct_cells": 0.1, "rank": 2, "n_cells": 10,
                                 "proj_type": "IT"},
                                {"rl_id": rl_id, "subclass_name": "Pvalb", "pct_cells": 0.1, "rank": 3, "n_cells": 10,
                                 "proj_type": "INT"}
                            ])
                        elif layer == "L5":
                            rel_rows.extend([
                                {"rl_id": rl_id, "subclass_name": "L5_IT", "pct_cells": 0.4, "rank": 1, "n_cells": 40,
                                 "proj_type": "IT"},
                                {"rl_id": rl_id, "subclass_name": "L5_ET", "pct_cells": 0.3, "rank": 2, "n_cells": 30,
                                 "proj_type": "ET"},
                                {"rl_id": rl_id, "subclass_name": "L5_NP", "pct_cells": 0.1, "rank": 3, "n_cells": 10,
                                 "proj_type": "NP"}
                            ])
                        elif layer == "L6":
                            rel_rows.extend([
                                {"rl_id": rl_id, "subclass_name": "L6_IT", "pct_cells": 0.3, "rank": 1, "n_cells": 30,
                                 "proj_type": "IT"},
                                {"rl_id": rl_id, "subclass_name": "L6_CT", "pct_cells": 0.4, "rank": 2, "n_cells": 40,
                                 "proj_type": "CT"},
                                {"rl_id": rl_id, "subclass_name": "Sst", "pct_cells": 0.1, "rank": 3, "n_cells": 10,
                                 "proj_type": "INT"}
                            ])
                        else:
                            rel_rows.extend([
                                {"rl_id": rl_id, "subclass_name": f"{layer}_Pyr", "pct_cells": 0.5, "rank": 1,
                                 "n_cells": 50, "proj_type": "IT"},
                                {"rl_id": rl_id, "subclass_name": "Pvalb", "pct_cells": 0.1, "rank": 2, "n_cells": 10,
                                 "proj_type": "INT"},
                                {"rl_id": rl_id, "subclass_name": "Sst", "pct_cells": 0.1, "rank": 3, "n_cells": 10,
                                 "proj_type": "INT"}
                            ])
                    elif rel_type == 'cluster':
                        # 集群关系 - 每层3个示例集群
                        rel_rows.extend([
                            {"rl_id": rl_id, "cluster_name": f"{layer}_Cluster1", "pct_cells": 0.4, "rank": 1,
                             "n_cells": 40},
                            {"rl_id": rl_id, "cluster_name": f"{layer}_Cluster2", "pct_cells": 0.3, "rank": 2,
                             "n_cells": 30},
                            {"rl_id": rl_id, "cluster_name": f"{layer}_Cluster3", "pct_cells": 0.1, "rank": 3,
                             "n_cells": 10}
                        ])

                # 保存关系数据
                rel_file = os.path.join(region_dir, f"has_{rel_type}.csv")
                pd.DataFrame(rel_rows).to_csv(rel_file, index=False)
                self.logger.info(f"保存{rel_type}关系到: {rel_file}")

            # 4. 更新并保存区域摘要
            region_data.update({
                "processed_date": self._get_timestamp(),
                "layers_processed": len(layers),
                "created_files": [
                    "region_layer_props.csv",
                    "has_class.csv",
                    "has_subclass.csv",
                    "has_cluster.csv"
                ]
            })

            # 保存区域摘要
            summary_file = os.path.join(region_dir, "region_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(region_data, f, indent=2)

            self.logger.info(f"脑区{region}处理完成")
            return True

        except Exception as e:
            self.logger.error(f"处理脑区{region}时出错: {e}", exc_info=True)
            return False
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
                filepath = rf"Z:\SEU-ALLEN\Users\Sujun\gene\cell_metadata_with_cluster_annotation_{i}.csv"
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
                    filepath = rf"Z:\SEU-ALLEN\Users\Sujun\gene\ccf_coordinates_{i}.csv"
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
                    log2_file = rf"Z:\SEU-ALLEN\Users\Sujun\gene\Zhuang-ABCA-{i}-log2.h5ad"
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

    def discover_all_regions(self):
        """
        通过坐标文件和CCF参考图谱发现所有可用的脑区
        """
        self.logger.info("开始从坐标和CCF参考图谱发现脑区...")
        all_regions = set()

        # ===============================================
        # 1. 加载CCF参考图谱和区域层次结构
        # ===============================================

        # 查找可能的参考图谱文件
        annotation_files = [
            os.path.join(r"D:\NeuroXiv", "annotation_25.nrrd"),
            os.path.join(r"D:\NeuroXiv", "annotation.nrrd"),
            os.path.join(r"D:\NeuroXiv", "ccf", "annotation_25.nrrd")
        ]

        # 查找可能的区域层次结构文件
        hierarchy_files = [
            os.path.join(r"D:\NeuroXiv", "ccf_2017.json"),
            os.path.join(r"D:\NeuroXiv", "tree.json"),
            os.path.join(r"D:\NeuroXiv", "ccf", "ccf_2017.json")
        ]

        # 查找第一个存在的文件
        annotation_file = next((f for f in annotation_files if os.path.exists(f)), None)
        hierarchy_file = next((f for f in hierarchy_files if os.path.exists(f)), None)

        if annotation_file is None:
            self.logger.warning("找不到CCF参考图谱文件")
            annotation_data = None
            voxel_size = None
        else:
            try:
                import nrrd
                self.logger.info(f"加载CCF参考图谱: {annotation_file}")
                annotation_data, header = nrrd.read(annotation_file)
                annotation_data = annotation_data.astype(np.uint32)

                # 获取体素尺寸 (μm)
                if "space directions" in header:
                    voxel_size = np.diag(header["space directions"])[:3].astype(float)
                    self.logger.info(f"CCF参考图谱体素尺寸: {voxel_size} μm")
                else:
                    voxel_size = np.array([25.0, 25.0, 25.0])  # 默认25μm
                    self.logger.warning(f"无法从头信息获取体素尺寸，使用默认值: {voxel_size} μm")

                # 输出图谱形状信息
                self.logger.info(f"CCF参考图谱形状: {annotation_data.shape}")
                self.logger.info(
                    f"CCF参考图谱范围 (μm): X[0-{annotation_data.shape[0] * voxel_size[0]:.0f}], Y[0-{annotation_data.shape[1] * voxel_size[1]:.0f}], Z[0-{annotation_data.shape[2] * voxel_size[2]:.0f}]")
            except Exception as e:
                self.logger.error(f"加载CCF参考图谱失败: {e}")
                annotation_data = None
                voxel_size = None

        # 加载区域层次结构
        if hierarchy_file is None:
            self.logger.warning("找不到区域层次结构文件")
            region_meta = {}
        else:
            try:
                self.logger.info(f"加载区域层次结构: {hierarchy_file}")
                with open(hierarchy_file, "r", encoding="utf-8") as f:
                    hierarchy_data = json.load(f)

                # 创建区域ID到信息的映射
                region_meta = {}

                # 递归遍历树形结构
                def traverse_tree(node):
                    if isinstance(node, dict):
                        if "id" in node:
                            region_meta[int(node["id"])] = node
                        if "children" in node and isinstance(node["children"], list):
                            for child in node["children"]:
                                traverse_tree(child)

                # 处理可能的不同格式
                if isinstance(hierarchy_data, list):
                    for item in hierarchy_data:
                        traverse_tree(item)
                elif isinstance(hierarchy_data, dict):
                    traverse_tree(hierarchy_data)

                self.logger.info(f"加载了{len(region_meta)}个区域元数据")
            except Exception as e:
                self.logger.error(f"加载区域层次结构失败: {e}")
                region_meta = {}

        # ===============================================
        # 2. 查找并处理坐标文件
        # ===============================================

        # 搜索坐标文件
        coord_files = []
        for i in range(1, 10):
            file_path = os.path.join(self.data_dir, f"ccf_coordinates_{i}.csv")
            if os.path.exists(file_path):
                coord_files.append(file_path)

        if not coord_files:
            self.logger.warning("未找到包含坐标的文件")
        else:
            # 处理每个坐标文件
            for file_path in coord_files:
                self.logger.info(f"处理坐标文件: {file_path}")
                try:
                    coords_df = pd.read_csv(file_path)

                    # 检查数据大小
                    self.logger.info(f"坐标数据形状: {coords_df.shape}")

                    # 识别坐标列
                    x_cols = [col for col in coords_df.columns if col.lower() in ['x', 'x_ccf', 'x_position', 'pos_x']]
                    y_cols = [col for col in coords_df.columns if col.lower() in ['y', 'y_ccf', 'y_position', 'pos_y']]
                    z_cols = [col for col in coords_df.columns if col.lower() in ['z', 'z_ccf', 'z_position', 'pos_z']]

                    if x_cols and y_cols and z_cols:
                        x_col, y_col, z_col = x_cols[0], y_cols[0], z_cols[0]
                        self.logger.info(f"使用坐标列: {x_col}, {y_col}, {z_col}")

                        # 检查坐标范围
                        x_range = (coords_df[x_col].min(), coords_df[x_col].max())
                        y_range = (coords_df[y_col].min(), coords_df[y_col].max())
                        z_range = (coords_df[z_col].min(), coords_df[z_col].max())
                        self.logger.info(f"坐标范围 - X: {x_range}, Y: {y_range}, Z: {z_range}")

                        # 判断坐标单位和转换因子
                        scale_factor = 1.0
                        if x_range[1] < 20 and y_range[1] < 20 and z_range[1] < 20:
                            scale_factor = 1000.0  # mm to μm
                            self.logger.info("检测到毫米坐标，应用转换因子: 1000")
                        elif x_range[1] < 200 and y_range[1] < 200 and z_range[1] < 200:
                            scale_factor = 100.0  # 0.1mm to μm
                            self.logger.info("检测到0.1毫米坐标，应用转换因子: 100")
                        else:
                            self.logger.info("坐标似乎已经是微米单位")

                        # 如果有参考图谱和元数据，查询区域ID和名称
                        if annotation_data is not None and voxel_size is not None:
                            # 创建一个采样的数据子集来避免处理所有数据
                            sample_size = min(10000, len(coords_df))
                            sample_indices = np.random.choice(len(coords_df), sample_size, replace=False)
                            sample_df = coords_df.iloc[sample_indices]

                            region_ids = []

                            # 处理采样的坐标
                            for _, row in sample_df.iterrows():
                                try:
                                    # 获取物理坐标 (μm) - 应用转换因子
                                    x = float(row[x_col]) * scale_factor
                                    y = float(row[y_col]) * scale_factor
                                    z = float(row[z_col]) * scale_factor

                                    # 转换为体素坐标
                                    # CCF v3使用PIR坐标系统（Posterior-Inferior-Right）
                                    # 可能需要调整坐标映射
                                    vx = int(x / voxel_size[0])
                                    vy = int(y / voxel_size[1])
                                    vz = int(z / voxel_size[2])

                                    # 确保坐标在有效范围内
                                    if (0 <= vx < annotation_data.shape[0] and
                                            0 <= vy < annotation_data.shape[1] and
                                            0 <= vz < annotation_data.shape[2]):

                                        # 获取区域ID
                                        region_id = int(annotation_data[vx, vy, vz])
                                        if region_id > 0:  # 跳过背景（ID=0）
                                            region_ids.append(region_id)
                                except Exception as e:
                                    continue

                            # 获取唯一区域ID
                            unique_region_ids = set(region_ids)
                            self.logger.info(f"从{sample_size}个采样点中找到{len(unique_region_ids)}个唯一区域ID")

                            # 显示前10个最常见的区域ID
                            if region_ids:
                                from collections import Counter
                                region_counts = Counter(region_ids)
                                top_regions = region_counts.most_common(10)
                                self.logger.info("最常见的区域ID:")
                                for rid, count in top_regions:
                                    if rid in region_meta:
                                        meta = region_meta[rid]
                                        name = meta.get("acronym", meta.get("name", f"ID_{rid}"))
                                        self.logger.info(f"  {rid}: {name} ({count}次)")

                            # 将ID映射到名称
                            for region_id in unique_region_ids:
                                if region_id in region_meta:
                                    meta = region_meta[region_id]
                                    # 优先使用acronym（简称），其次使用name
                                    region_name = meta.get("acronym", meta.get("name", f"Region_{region_id}"))
                                    if region_name and isinstance(region_name, str):
                                        all_regions.add(region_name)
                    else:
                        self.logger.warning(f"在{file_path}中未找到坐标列")
                except Exception as e:
                    self.logger.error(f"处理坐标文件{file_path}时出错: {e}")

        # ===============================================
        # 3. 处理特殊情况和返回结果
        # ===============================================

        # 如果没有找到区域，使用默认的皮层区域
        if not all_regions:
            self.logger.warning("未能识别任何脑区，使用预定义的主要皮层区域")
            all_regions = {'MOp', 'MOs', 'SSp', 'SSs', 'ACA', 'PL', 'ILA', 'ORB',
                           'AI', 'RSP', 'PTLp', 'VISp', 'VISl', 'VISal', 'VISam',
                           'VISpm', 'TEa', 'PERI', 'ECT'}

        # 过滤掉非皮层区域
        cortical_prefixes = ['MO', 'SS', 'VIS', 'ACA', 'AI', 'RSP', 'PTL', 'TEa', 'PERI', 'ECT', 'PL', 'ILA', 'ORB']
        if len(all_regions) > 100:  # 如果区域太多，只保留皮层区域
            self.logger.info(f"筛选皮层区域（从{len(all_regions)}个区域中）")
            cortical_regions = {r for r in all_regions if any(r.startswith(p) for p in cortical_prefixes)}
            if cortical_regions:
                all_regions = cortical_regions
                self.logger.info(f"保留{len(all_regions)}个皮层区域")

        # 将结果排序并返回
        result = sorted(list(all_regions))
        self.logger.info(f"发现{len(result)}个脑区: {', '.join(result[:10])}" +
                         (f"... 等{len(result) - 10}个" if len(result) > 10 else ""))
        return result

    def transform_merfish_to_ccf(self, x, y, z, scale_factor=1.0):
        """
        将MERFISH坐标转换为CCF v3坐标

        Args:
            x, y, z: MERFISH坐标
            scale_factor: 单位转换因子（例如1000表示mm到μm）

        Returns:
            ccf_x, ccf_y, ccf_z: CCF坐标（微米）
        """
        # 应用缩放因子
        x_um = x * scale_factor
        y_um = y * scale_factor
        z_um = z * scale_factor

        # MERFISH和CCF可能使用不同的坐标系统
        # CCF v3使用PIR（Posterior-Inferior-Right）坐标系
        # 可能需要调整轴的映射或方向

        # 默认映射（可能需要根据实际数据调整）
        ccf_x = x_um
        ccf_y = y_um
        ccf_z = z_um

        return ccf_x, ccf_y, ccf_z

    def ccf_to_voxel(self, ccf_x, ccf_y, ccf_z, voxel_size):
        """
        将CCF物理坐标（微米）转换为体素坐标

        Args:
            ccf_x, ccf_y, ccf_z: CCF坐标（微米）
            voxel_size: 体素尺寸数组 [vx, vy, vz]（微米）

        Returns:
            vx, vy, vz: 体素坐标（整数）
        """
        vx = int(ccf_x / voxel_size[0])
        vy = int(ccf_y / voxel_size[1])
        vz = int(ccf_z / voxel_size[2])

        return vx, vy, vz

    def validate_ccf_coordinates(self, coords_df, x_col, y_col, z_col, annotation_shape, voxel_size):
        """
        验证坐标转换是否正确

        Args:
            coords_df: 坐标数据框
            x_col, y_col, z_col: 坐标列名
            annotation_shape: 注释图谱的形状
            voxel_size: 体素尺寸

        Returns:
            scale_factor: 推荐的缩放因子
            axis_mapping: 推荐的轴映射
        """
        # 计算坐标范围
        x_range = (coords_df[x_col].min(), coords_df[x_col].max())
        y_range = (coords_df[y_col].min(), coords_df[y_col].max())
        z_range = (coords_df[z_col].min(), coords_df[z_col].max())

        # CCF v3的物理范围（微米）
        ccf_x_max = annotation_shape[0] * voxel_size[0]
        ccf_y_max = annotation_shape[1] * voxel_size[1]
        ccf_z_max = annotation_shape[2] * voxel_size[2]

        self.logger.info(f"CCF物理范围 (μm): X[0-{ccf_x_max:.0f}], Y[0-{ccf_y_max:.0f}], Z[0-{ccf_z_max:.0f}]")
        self.logger.info(f"输入坐标范围: X{x_range}, Y{y_range}, Z{z_range}")

        # 估算缩放因子
        scale_factors = []

        # 检查每个轴
        for coord_range, ccf_max in [(x_range, ccf_x_max), (y_range, ccf_y_max), (z_range, ccf_z_max)]:
            if coord_range[1] > 0:
                # 估算需要的缩放因子使坐标落在CCF范围内
                estimated_scale = ccf_max / (coord_range[1] * 1.2)  # 留20%余量

                # 找到最接近的标准缩放因子
                standard_scales = [1, 10, 100, 1000, 10000]
                closest_scale = min(standard_scales, key=lambda x: abs(x - estimated_scale))
                scale_factors.append(closest_scale)

        # 使用最常见的缩放因子
        from collections import Counter
        scale_counter = Counter(scale_factors)
        scale_factor = scale_counter.most_common(1)[0][0] if scale_factors else 1.0

        self.logger.info(f"推荐缩放因子: {scale_factor}")

        # 测试一些采样点
        sample_size = min(100, len(coords_df))
        sample_df = coords_df.sample(n=sample_size)

        valid_count = 0
        for _, row in sample_df.iterrows():
            x = row[x_col] * scale_factor
            y = row[y_col] * scale_factor
            z = row[z_col] * scale_factor

            vx = int(x / voxel_size[0])
            vy = int(y / voxel_size[1])
            vz = int(z / voxel_size[2])

            if (0 <= vx < annotation_shape[0] and
                    0 <= vy < annotation_shape[1] and
                    0 <= vz < annotation_shape[2]):
                valid_count += 1

        valid_ratio = valid_count / sample_size
        self.logger.info(f"验证结果: {valid_ratio:.1%}的采样点落在有效范围内")

        return scale_factor, None  # axis_mapping暂时返回None

    def detect_coordinate_system(self, coords_df, metadata_df=None):
        """
        检测坐标系统类型

        Args:
            coords_df: 坐标数据
            metadata_df: 元数据（可选）

        Returns:
            coord_system: 坐标系统类型 ('merfish', 'ccf', 'unknown')
            scale_factor: 推荐的缩放因子
        """
        # 检查列名
        columns = coords_df.columns.tolist()

        # MERFISH特征
        merfish_indicators = ['cell_id', 'gene', 'transcript_id']
        ccf_indicators = ['x_ccf', 'y_ccf', 'z_ccf', 'ccf_x', 'ccf_y', 'ccf_z']

        has_merfish = any(ind in columns for ind in merfish_indicators)
        has_ccf = any(ind in columns for ind in ccf_indicators)

        if has_ccf:
            return 'ccf', 1.0
        elif has_merfish:
            # MERFISH通常需要缩放
            return 'merfish', 1000.0
        else:
            # 基于坐标范围判断
            x_cols = [col for col in columns if col.lower() == 'x']
            if x_cols:
                x_max = coords_df[x_cols[0]].max()
                if x_max < 20:
                    return 'merfish', 1000.0  # 毫米
                elif x_max < 200:
                    return 'merfish', 100.0  # 0.1毫米
                elif x_max > 5000:
                    return 'ccf', 1.0  # 已经是微米

            return 'unknown', 1.0

    def _validate_coordinates(self, coords_df, x_col, y_col, z_col):
        """验证坐标数据并确定是否需要转换"""
        # 检查坐标范围来推断可能的坐标系统
        x_range = (coords_df[x_col].min(), coords_df[x_col].max())
        y_range = (coords_df[y_col].min(), coords_df[y_col].max())
        z_range = (coords_df[z_col].min(), coords_df[z_col].max())

        self.logger.info(f"坐标范围 - X: {x_range}, Y: {y_range}, Z: {z_range}")

        # 判断坐标是否已经是CCF单位（μm）或需要转换
        # CCF v3的坐标范围通常在几千到几万μm之间
        if x_range[1] < 100 and y_range[1] < 100 and z_range[1] < 100:
            self.logger.info("检测到MERFISH坐标（毫米或厘米），需要乘以1000转换为CCF v3单位(μm)")
            return 1000.0  # 转换因子
        elif x_range[1] < 1000 and y_range[1] < 1000 and z_range[1] < 1000:
            self.logger.info("检测到缩放的坐标，需要乘以转换因子匹配CCF v3")
            return 100.0  # 假设缩放了100倍
        else:
            self.logger.info("坐标似乎已经是CCF v3单位(μm)，无需转换")
            return 1.0  # 无需转换

    def calculate_cross_region_statistics(self):
        """计算跨区域统计分析，生成汇总数据"""
        self.logger.info("开始计算跨区域统计分析...")

        # 创建跨区域分析的输出目录
        cross_region_dir = os.path.join(self.output_dir, "cross_region_analysis")
        os.makedirs(cross_region_dir, exist_ok=True)

        try:
            # 1. 收集所有处理过的区域数据
            processed_regions = []
            region_data = {}

            # 遍历输出目录，查找已处理的区域数据
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path) and not item.startswith(
                        '.') and item != "cross_region_analysis" and item != "cache":
                    # 检查是否有区域摘要文件，用于确认是有效的区域目录
                    summary_file = os.path.join(item_path, "region_summary.json")
                    if os.path.exists(summary_file):
                        processed_regions.append(item)
                        try:
                            with open(summary_file, 'r') as f:
                                region_data[item] = json.load(f)
                        except Exception as e:
                            self.logger.warning(f"加载区域{item}数据失败: {e}")

            if not processed_regions:
                self.logger.warning("未找到已处理的区域数据，跳过跨区域分析")
                return

            self.logger.info(f"找到{len(processed_regions)}个已处理区域: {', '.join(processed_regions)}")

            # 2. 合并各区域的RegionLayer属性
            region_layer_props_files = [os.path.join(self.output_dir, region, "region_layer_props.csv")
                                        for region in processed_regions]
            combined_rl_props = []

            for file_path in region_layer_props_files:
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        combined_rl_props.append(df)
                    except Exception as e:
                        self.logger.warning(f"读取{file_path}失败: {e}")

            if combined_rl_props:
                # 合并所有区域的RegionLayer属性
                all_rl_props = pd.concat(combined_rl_props, ignore_index=True)

                # 保存合并后的RegionLayer属性
                output_file = os.path.join(self.output_dir, "region_layer_props.csv")
                all_rl_props.to_csv(output_file, index=False)
                self.logger.info(f"保存合并后的RegionLayer属性到: {output_file}")

            # 3. 类似处理各类转录组关系
            for rel_type in ['class', 'subclass', 'cluster']:
                rel_files = [os.path.join(self.output_dir, region, f"has_{rel_type}.csv")
                             for region in processed_regions]
                combined_data = []

                for file_path in rel_files:
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            combined_data.append(df)
                        except Exception as e:
                            self.logger.warning(f"读取{file_path}失败: {e}")

                if combined_data:
                    # 合并所有区域的关系
                    all_data = pd.concat(combined_data, ignore_index=True)

                    # 保存合并后的关系
                    output_file = os.path.join(self.output_dir, f"has_{rel_type}.csv")
                    all_data.to_csv(output_file, index=False)
                    self.logger.info(f"保存合并后的{rel_type}关系到: {output_file}")

                    # 特殊处理亚类-投射类型映射
                    if rel_type == 'subclass' and 'subclass_name' in all_data.columns and 'proj_type' in all_data.columns:
                        try:
                            # 提取每个亚类的投射类型
                            subclass_proj = all_data[['subclass_name', 'proj_type']].drop_duplicates()

                            # 保存亚类-投射类型映射
                            output_file = os.path.join(self.output_dir, "subclass_projtype.csv")
                            subclass_proj.to_csv(output_file, index=False)
                            self.logger.info(f"保存亚类-投射类型映射到: {output_file}")
                        except Exception as e:
                            self.logger.warning(f"创建亚类-投射类型映射失败: {e}")

            # 4. 创建跨区域分析摘要
            summary = {
                "processed_regions": processed_regions,
                "processed_regions_count": len(processed_regions),
                "cross_region_statistics": {
                    "region_layer_count": len(all_rl_props) if 'all_rl_props' in locals() else 0,
                    "has_class_count": len(all_data) if 'all_data' in locals() and rel_type == 'class' else 0,
                    "has_subclass_count": len(all_data) if 'all_data' in locals() and rel_type == 'subclass' else 0,
                    "has_cluster_count": len(all_data) if 'all_data' in locals() and rel_type == 'cluster' else 0
                },
                "timestamp": self._get_timestamp()
            }

            # 保存摘要
            summary_file = os.path.join(cross_region_dir, "cross_region_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"跨区域统计分析完成，摘要已保存到: {summary_file}")

        except Exception as e:
            self.logger.error(f"跨区域统计分析过程中出错: {e}", exc_info=True)

    def _get_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    parser = argparse.ArgumentParser(description='MERFISH数据集成工具')
    parser.add_argument('--regions', nargs='+', default=['ALL'],
                        help='要处理的脑区列表，使用"ALL"处理所有脑区')
    parser.add_argument('--output-dir', default='merfish_output', help='输出目录')
    parser.add_argument('--data-dir', default='.', help='数据目录')
    parser.add_argument('--job-id', default='', help='可选的作业ID后缀')
    parser.add_argument('--skip-cross-region', action='store_true',
                        help='跳过跨区域分析')
    args = parser.parse_args()

    try:
        # 初始化集成器
        integrator = MERFISHDataIntegration(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            job_id=args.job_id
        )

        # 确定要处理的区域
        if 'ALL' in args.regions or not args.regions:
            regions_to_process = integrator.discover_all_regions()
            if regions_to_process:
                logger.info(f"将处理所有可用脑区: {', '.join(regions_to_process)}")
            else:
                logger.warning("未找到任何可用脑区，处理终止")
                return
        else:
            regions_to_process = args.regions
            logger.info(f"将处理指定脑区: {', '.join(regions_to_process)}")

        # 处理每个区域
        for region in regions_to_process:
            try:
                logger.info(f"开始处理脑区: {region}")
                # 直接处理整个区域名，不要拆分
                integrator.process_region(region)
            except Exception as e:
                logger.error(f"处理脑区{region}时发生错误: {e}", exc_info=True)

        # 运行跨区域分析（除非被跳过）
        if not args.skip_cross_region:
            logger.info("开始跨区域分析...")
            integrator.calculate_cross_region_statistics()
            logger.info("跨区域分析完成")
        else:
            logger.info("跳过跨区域分析")

        logger.info("所有脑区处理完成")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()