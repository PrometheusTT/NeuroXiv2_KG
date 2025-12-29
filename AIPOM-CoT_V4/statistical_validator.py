"""
Statistical Validator - 真正的统计验证实现
============================================

实现手稿中声称的统计方法:
- Permutation-based p-values
- Benjamini-Hochberg FDR correction
- Bootstrap confidence intervals
- Cohen's d effect sizes

这些方法对应手稿Methods部分和Figure 5的统计验证

Author: Lijun
Date: 2025-01
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """统计检验结果"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    cohens_d: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    fdr_q: Optional[float] = None
    sample_size: int = 0
    is_significant: bool = False
    interpretation: str = ""

    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'cohens_d': self.cohens_d,
            'ci': self.confidence_interval,
            'fdr_q': self.fdr_q,
            'n': self.sample_size,
            'significant': self.is_significant,
            'interpretation': self.interpretation
        }


class StatisticalValidator:
    """
    统计验证器 - 实现手稿中声称的所有统计方法

    核心功能:
    1. permutation_test - 置换检验计算p值
    2. benjamini_hochberg - FDR校正
    3. bootstrap_ci - Bootstrap置信区间
    4. cohens_d - Cohen's d效应量
    5. comprehensive_test - 综合统计检验
    """

    def __init__(self, alpha: float = 0.05, n_permutations: int = 1000,
                 n_bootstrap: int = 1000, random_seed: int = 42):
        """
        初始化统计验证器

        Args:
            alpha: 显著性水平 (默认0.05)
            n_permutations: 置换次数 (默认1000)
            n_bootstrap: Bootstrap次数 (默认1000)
            random_seed: 随机种子
        """
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_seed)

    # ==================== Core Statistical Methods ====================

    def permutation_test(self, group1: np.ndarray, group2: np.ndarray,
                        statistic_func: str = 'mean_diff',
                        alternative: str = 'two-sided') -> StatisticalResult:
        """
        置换检验 - 手稿Methods声称的核心方法

        通过随机置换标签来构建null distribution，
        计算观察到的统计量在null distribution中的位置

        Args:
            group1: 第一组数据
            group2: 第二组数据
            statistic_func: 统计量类型 ('mean_diff', 'median_diff', 't_stat')
            alternative: 备择假设 ('two-sided', 'greater', 'less')

        Returns:
            StatisticalResult: 包含p值、效应量等
        """
        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        n1, n2 = len(group1), len(group2)
        combined = np.concatenate([group1, group2])

        # 计算观察统计量
        observed_stat = self._compute_statistic(group1, group2, statistic_func)

        # 置换检验
        perm_stats = np.zeros(self.n_permutations)
        for i in range(self.n_permutations):
            self.rng.shuffle(combined)
            perm_g1 = combined[:n1]
            perm_g2 = combined[n1:]
            perm_stats[i] = self._compute_statistic(perm_g1, perm_g2, statistic_func)

        # 计算p值
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        elif alternative == 'greater':
            p_value = np.mean(perm_stats >= observed_stat)
        else:  # less
            p_value = np.mean(perm_stats <= observed_stat)

        # 计算效应量
        cohens_d_val = self.cohens_d(group1, group2)

        return StatisticalResult(
            test_name='permutation_test',
            statistic=observed_stat,
            p_value=p_value,
            effect_size=cohens_d_val,
            cohens_d=cohens_d_val,
            sample_size=n1 + n2,
            is_significant=p_value < self.alpha,
            interpretation=self._interpret_effect_size(cohens_d_val)
        )

    def _compute_statistic(self, g1: np.ndarray, g2: np.ndarray,
                          stat_type: str) -> float:
        """计算统计量"""
        if stat_type == 'mean_diff':
            return np.mean(g1) - np.mean(g2)
        elif stat_type == 'median_diff':
            return np.median(g1) - np.median(g2)
        elif stat_type == 't_stat':
            # Welch's t-statistic
            n1, n2 = len(g1), len(g2)
            var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
            se = np.sqrt(var1/n1 + var2/n2)
            if se == 0:
                return 0.0
            return (np.mean(g1) - np.mean(g2)) / se
        else:
            return np.mean(g1) - np.mean(g2)

    def benjamini_hochberg(self, p_values: List[float]) -> List[float]:
        """
        Benjamini-Hochberg FDR校正 - 手稿Methods声称的多重比较校正

        控制False Discovery Rate，适用于多重假设检验

        Args:
            p_values: 原始p值列表

        Returns:
            q_values: FDR校正后的q值列表
        """
        p_values = np.asarray(p_values)
        n = len(p_values)

        if n == 0:
            return []

        # 排序索引
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # 计算BH校正后的q值
        q_values = np.zeros(n)
        cummin = 1.0

        for i in range(n - 1, -1, -1):
            # q_i = min(p_i * n / rank, q_{i+1})
            rank = i + 1
            q = sorted_p[i] * n / rank
            cummin = min(cummin, q)
            q_values[sorted_indices[i]] = min(1.0, cummin)

        return q_values.tolist()

    def bootstrap_ci(self, data: np.ndarray, statistic: str = 'mean',
                    confidence: float = 0.95) -> Tuple[float, float]:
        """
        Bootstrap置信区间 - 手稿声称的置信区间估计方法

        通过重采样构建统计量的分布，计算置信区间

        Args:
            data: 数据数组
            statistic: 统计量类型 ('mean', 'median', 'std')
            confidence: 置信水平 (默认0.95)

        Returns:
            (lower, upper): 置信区间
        """
        data = np.asarray(data).flatten()
        n = len(data)

        if n == 0:
            return (np.nan, np.nan)

        # Bootstrap重采样
        boot_stats = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            boot_sample = self.rng.choice(data, size=n, replace=True)
            if statistic == 'mean':
                boot_stats[i] = np.mean(boot_sample)
            elif statistic == 'median':
                boot_stats[i] = np.median(boot_sample)
            elif statistic == 'std':
                boot_stats[i] = np.std(boot_sample, ddof=1)
            else:
                boot_stats[i] = np.mean(boot_sample)

        # 计算百分位置信区间
        alpha = 1 - confidence
        lower = np.percentile(boot_stats, 100 * alpha / 2)
        upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Cohen's d效应量 - 手稿声称的效应量度量

        衡量两组之间差异的标准化大小

        Args:
            group1: 第一组数据
            group2: 第二组数据

        Returns:
            Cohen's d值
        """
        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        n1, n2 = len(group1), len(group2)

        if n1 < 2 or n2 < 2:
            return 0.0

        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """解释效应量大小 (Cohen's conventions)"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    # ==================== Comprehensive Testing ====================

    def comprehensive_test(self, group1: np.ndarray, group2: np.ndarray,
                          test_name: str = "") -> StatisticalResult:
        """
        综合统计检验 - 包含所有手稿声称的统计方法

        执行:
        1. Permutation test获取p值
        2. Bootstrap CI
        3. Cohen's d效应量

        Args:
            group1: 第一组数据
            group2: 第二组数据
            test_name: 检验名称

        Returns:
            包含所有统计指标的结果
        """
        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        # 1. Permutation test
        perm_result = self.permutation_test(group1, group2)

        # 2. Bootstrap CI for mean difference
        diff = group1.mean() - group2.mean() if len(group1) > 0 and len(group2) > 0 else 0

        # Bootstrap CI for the difference
        combined_for_ci = np.concatenate([group1 - group1.mean(), group2 - group2.mean()])
        boot_diffs = np.zeros(self.n_bootstrap)
        n1, n2 = len(group1), len(group2)

        for i in range(self.n_bootstrap):
            boot1 = self.rng.choice(group1, size=n1, replace=True)
            boot2 = self.rng.choice(group2, size=n2, replace=True)
            boot_diffs[i] = np.mean(boot1) - np.mean(boot2)

        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)

        # 3. Cohen's d
        d = self.cohens_d(group1, group2)

        return StatisticalResult(
            test_name=test_name or 'comprehensive_test',
            statistic=diff,
            p_value=perm_result.p_value,
            effect_size=d,
            cohens_d=d,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=n1 + n2,
            is_significant=perm_result.p_value < self.alpha,
            interpretation=self._interpret_effect_size(d)
        )

    # ==================== Multi-group Comparisons ====================

    def pairwise_comparisons(self, groups: Dict[str, np.ndarray],
                            apply_fdr: bool = True) -> Dict[str, StatisticalResult]:
        """
        多组两两比较 + FDR校正

        Args:
            groups: 组名到数据的映射
            apply_fdr: 是否应用FDR校正

        Returns:
            比较对到结果的映射
        """
        group_names = list(groups.keys())
        n_groups = len(group_names)

        results = {}
        p_values = []
        comparison_keys = []

        # 执行所有两两比较
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                name1, name2 = group_names[i], group_names[j]
                key = f"{name1}_vs_{name2}"

                result = self.comprehensive_test(
                    groups[name1],
                    groups[name2],
                    test_name=key
                )

                results[key] = result
                p_values.append(result.p_value)
                comparison_keys.append(key)

        # 应用FDR校正
        if apply_fdr and p_values:
            q_values = self.benjamini_hochberg(p_values)
            for key, q in zip(comparison_keys, q_values):
                results[key].fdr_q = q
                # 更新显著性判断（基于FDR）
                results[key].is_significant = q < self.alpha

        return results

    def validate_fingerprint_similarity(self,
                                        similarity_matrix: np.ndarray,
                                        expected_pattern: str = 'block_diagonal') -> Dict:
        """
        验证指纹相似性矩阵的统计显著性

        用于验证Figure 4中的tri-modal fingerprint相似性分析

        Args:
            similarity_matrix: 相似性矩阵 (NxN)
            expected_pattern: 期望的模式类型

        Returns:
            验证结果字典
        """
        n = similarity_matrix.shape[0]

        if n < 2:
            return {'valid': False, 'reason': 'Matrix too small'}

        # 对角线元素（自相似）
        diagonal = np.diag(similarity_matrix)

        # 非对角线元素（跨区域相似）
        off_diagonal = similarity_matrix[~np.eye(n, dtype=bool)]

        # 检验：对角线应显著高于非对角线
        if len(off_diagonal) > 0:
            result = self.permutation_test(diagonal, off_diagonal, alternative='greater')

            return {
                'valid': True,
                'diagonal_mean': float(np.mean(diagonal)),
                'off_diagonal_mean': float(np.mean(off_diagonal)),
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'is_significant': result.is_significant,
                'interpretation': 'Regions show distinct fingerprints' if result.is_significant
                                 else 'Fingerprints not significantly distinct'
            }

        return {'valid': False, 'reason': 'No off-diagonal elements'}

    # ==================== Specialized Methods for Neuroscience ====================

    def compare_modality_contributions(self,
                                       molecular: np.ndarray,
                                       morphological: np.ndarray,
                                       projection: np.ndarray) -> Dict:
        """
        比较三种模态的贡献度

        用于Result 4的tri-modal分析

        Args:
            molecular: 分子模态相似度
            morphological: 形态模态相似度
            projection: 投射模态相似度

        Returns:
            模态比较结果
        """
        modalities = {
            'molecular': np.asarray(molecular).flatten(),
            'morphological': np.asarray(morphological).flatten(),
            'projection': np.asarray(projection).flatten()
        }

        # 两两比较
        comparisons = self.pairwise_comparisons(modalities, apply_fdr=True)

        # 计算每个模态的均值和CI
        modality_stats = {}
        for name, data in modalities.items():
            if len(data) > 0:
                ci = self.bootstrap_ci(data, 'mean')
                modality_stats[name] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'n': len(data)
                }

        return {
            'modality_stats': modality_stats,
            'pairwise_comparisons': {k: v.to_dict() for k, v in comparisons.items()},
            'dominant_modality': max(modality_stats.keys(),
                                    key=lambda x: modality_stats[x]['mean'])
                                if modality_stats else None
        }

    def cross_modal_mismatch_test(self,
                                  sim_matrix_1: np.ndarray,
                                  sim_matrix_2: np.ndarray,
                                  region_names: List[str] = None) -> Dict:
        """
        检测跨模态不匹配

        找出在不同模态下表现差异显著的区域对

        Args:
            sim_matrix_1: 第一模态相似性矩阵
            sim_matrix_2: 第二模态相似性矩阵
            region_names: 区域名称列表

        Returns:
            不匹配分析结果
        """
        diff_matrix = sim_matrix_1 - sim_matrix_2
        n = diff_matrix.shape[0]

        if region_names is None:
            region_names = [f"Region_{i}" for i in range(n)]

        # 找出显著不匹配的区域对
        mismatches = []
        threshold = 2 * np.std(diff_matrix)  # 2个标准差作为阈值

        for i in range(n):
            for j in range(i + 1, n):
                diff = diff_matrix[i, j]
                if abs(diff) > threshold:
                    mismatches.append({
                        'region_1': region_names[i],
                        'region_2': region_names[j],
                        'difference': float(diff),
                        'z_score': float(diff / np.std(diff_matrix)) if np.std(diff_matrix) > 0 else 0
                    })

        # 按差异大小排序
        mismatches.sort(key=lambda x: abs(x['difference']), reverse=True)

        return {
            'n_mismatches': len(mismatches),
            'top_mismatches': mismatches[:10],  # 前10个
            'mean_diff': float(np.mean(np.abs(diff_matrix))),
            'std_diff': float(np.std(diff_matrix)),
            'correlation': float(np.corrcoef(sim_matrix_1.flatten(),
                                            sim_matrix_2.flatten())[0, 1])
                          if sim_matrix_1.size > 0 else 0.0
        }


class BenchmarkStatisticalValidator:
    """
    Benchmark统计验证器 - 用于Figure 5的方法比较

    确保AIPOM-CoT vs baselines的比较具有统计显著性
    """

    def __init__(self, n_bootstrap: int = 1000, alpha: float = 0.05):
        self.validator = StatisticalValidator(
            alpha=alpha,
            n_permutations=1000,
            n_bootstrap=n_bootstrap
        )
        self.alpha = alpha

    def validate_benchmark_results(self,
                                   results: Dict[str, List[float]]) -> Dict:
        """
        验证benchmark结果的统计显著性

        Args:
            results: 方法名到得分列表的映射
                    例如: {'AIPOM-CoT': [0.9, 0.85, ...], 'RAG': [0.7, ...], ...}

        Returns:
            完整的统计验证结果
        """
        method_names = list(results.keys())

        # 1. 每个方法的描述统计
        method_stats = {}
        for name, scores in results.items():
            scores = np.asarray(scores)
            ci = self.validator.bootstrap_ci(scores, 'mean')
            method_stats[name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'n': len(scores)
            }

        # 2. 两两比较 + FDR校正
        comparisons = self.validator.pairwise_comparisons(
            {k: np.asarray(v) for k, v in results.items()},
            apply_fdr=True
        )

        # 3. 找出最佳方法
        best_method = max(method_stats.keys(),
                         key=lambda x: method_stats[x]['mean'])

        # 4. 检验最佳方法是否显著优于其他方法
        significantly_better_than = []
        for name in method_names:
            if name != best_method:
                key1 = f"{best_method}_vs_{name}"
                key2 = f"{name}_vs_{best_method}"
                key = key1 if key1 in comparisons else key2

                if key in comparisons:
                    result = comparisons[key]
                    # 检查方向：best_method是否显著高于name
                    if result.is_significant:
                        mean_best = method_stats[best_method]['mean']
                        mean_other = method_stats[name]['mean']
                        if mean_best > mean_other:
                            significantly_better_than.append(name)

        return {
            'method_statistics': method_stats,
            'pairwise_comparisons': {k: v.to_dict() for k, v in comparisons.items()},
            'best_method': best_method,
            'significantly_better_than': significantly_better_than,
            'all_comparisons_significant': len(significantly_better_than) == len(method_names) - 1
        }

    def validate_capability_scores(self,
                                   capability_scores: Dict[str, Dict[str, List[float]]]) -> Dict:
        """
        验证各能力维度得分的统计显著性

        Args:
            capability_scores: 嵌套字典
                              {方法: {能力: [得分列表]}}

        Returns:
            各能力维度的比较结果
        """
        capabilities = ['think', 'plan', 'act', 'reflect']
        results = {}

        for cap in capabilities:
            # 收集该能力下各方法的得分
            cap_data = {}
            for method, caps in capability_scores.items():
                if cap in caps and len(caps[cap]) > 0:
                    cap_data[method] = np.asarray(caps[cap])

            if len(cap_data) >= 2:
                results[cap] = self.validator.pairwise_comparisons(cap_data, apply_fdr=True)
                results[cap] = {k: v.to_dict() for k, v in results[cap].items()}

        return results


# ==================== Static Method Wrappers for benchmark.py Compatibility ====================

class StatisticalValidatorStatic:
    """
    静态方法版本的统计验证器
    兼容benchmark.py的调用方式
    """

    @staticmethod
    def permutation_test(group1: List[float], group2: List[float],
                        n_permutations: int = 1000) -> Dict:
        """
        置换检验（静态方法版本）

        Args:
            group1: 第一组数据
            group2: 第二组数据
            n_permutations: 置换次数

        Returns:
            包含统计结果的字典
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        n1, n2 = len(group1), len(group2)
        combined = np.concatenate([group1, group2])

        # 观察到的差异
        observed_diff = np.mean(group1) - np.mean(group2)

        # 置换检验
        rng = np.random.RandomState(42)
        perm_diffs = np.zeros(n_permutations)

        for i in range(n_permutations):
            rng.shuffle(combined)
            perm_diffs[i] = np.mean(combined[:n1]) - np.mean(combined[n1:])

        # p值（双侧）
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

        # Cohen's d
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

        return {
            'mean1': float(np.mean(group1)),
            'mean2': float(np.mean(group2)),
            'observed_diff': float(observed_diff),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < 0.05,
            'n1': n1,
            'n2': n2
        }

    @staticmethod
    def bootstrap_ci(data: List[float], confidence: float = 0.95,
                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Bootstrap置信区间（静态方法版本）

        Args:
            data: 数据
            confidence: 置信水平
            n_bootstrap: 重采样次数

        Returns:
            (lower, upper) 置信区间
        """
        data = np.asarray(data)
        n = len(data)

        if n == 0:
            return (np.nan, np.nan)

        rng = np.random.RandomState(42)
        boot_means = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            boot_sample = rng.choice(data, size=n, replace=True)
            boot_means[i] = np.mean(boot_sample)

        alpha = 1 - confidence
        lower = np.percentile(boot_means, 100 * alpha / 2)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    @staticmethod
    def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
        """
        Benjamini-Hochberg FDR校正（静态方法版本）

        Args:
            p_values: 原始p值列表
            alpha: 显著性水平

        Returns:
            (q_values, significant): 校正后q值和显著性判断
        """
        p_values = np.asarray(p_values)
        n = len(p_values)

        if n == 0:
            return [], []

        # 排序
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # BH校正
        q_values = np.zeros(n)
        cummin = 1.0

        for i in range(n - 1, -1, -1):
            rank = i + 1
            q = sorted_p[i] * n / rank
            cummin = min(cummin, q)
            q_values[sorted_indices[i]] = min(1.0, cummin)

        significant = [q < alpha for q in q_values]

        return q_values.tolist(), significant


# 为了兼容 benchmark.py，给 StatisticalValidator 添加静态方法
StatisticalValidator.permutation_test_static = staticmethod(StatisticalValidatorStatic.permutation_test)
StatisticalValidator.bootstrap_ci_static = staticmethod(StatisticalValidatorStatic.bootstrap_ci)
StatisticalValidator.fdr_correction_static = staticmethod(StatisticalValidatorStatic.fdr_correction)

# 直接作为类方法（benchmark.py 的调用方式）
# 覆盖为静态方法版本
_original_validator = StatisticalValidator

class StatisticalValidator:
    """
    统计验证器 - 兼容版本
    同时支持实例方法和静态方法调用
    """

    def __init__(self, alpha: float = 0.05, n_permutations: int = 1000,
                 n_bootstrap: int = 1000, random_seed: int = 42):
        self._validator = _original_validator(alpha, n_permutations, n_bootstrap, random_seed)

    def __getattr__(self, name):
        return getattr(self._validator, name)

    # 静态方法 - 兼容 benchmark.py
    @staticmethod
    def permutation_test(group1, group2, n_permutations: int = 1000):
        """静态版本的置换检验"""
        return StatisticalValidatorStatic.permutation_test(group1, group2, n_permutations)

    @staticmethod
    def bootstrap_ci(data, confidence: float = 0.95, n_bootstrap: int = 1000):
        """静态版本的Bootstrap CI"""
        return StatisticalValidatorStatic.bootstrap_ci(data, confidence, n_bootstrap)

    @staticmethod
    def fdr_correction(p_values, alpha: float = 0.05):
        """静态版本的FDR校正"""
        return StatisticalValidatorStatic.fdr_correction(p_values, alpha)


# ==================== Export ====================

__all__ = [
    'StatisticalResult',
    'StatisticalValidator',
    'StatisticalValidatorStatic',
    'BenchmarkStatisticalValidator'
]