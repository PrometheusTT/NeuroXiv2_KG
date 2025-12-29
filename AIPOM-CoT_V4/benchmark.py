#!/usr/bin/env python3
"""
NeuroXiv-Agent Benchmark Suite
===============================
完整的Benchmark测试框架

功能：
1. 运行所有方法（NeuroXiv-Agent + Baselines）
2. 能力评估（Think/Plan/Reflect/Act）
3. 正确性评估
4. 统计显著性检验
5. 结果可视化

使用方法：
    # Mock模式快速测试
    python benchmark.py --mode mock --questions 5
    
    # 完整测试（需要数据库和API）
    python benchmark.py --mode full --output results/

Author: Lijun
Date: 2025-01
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Imports ====================

from questions import (
    ALL_QUESTIONS, 
    get_questions_by_level,
    get_balanced_sample,
    get_question_stats
)
from evaluator import (
    CapabilityEvaluator,
    StatisticalValidator,
    EvaluationResult,
    CAPABILITY_WEIGHTS
)
from baselines import (
    DirectLLMAgent,
    ReActAgent,
    SimpleRAGAgent,
    create_baseline_agents
)
from agent import NeuroXivAgent, MockExecutor, MockLLMClient


# ==================== Benchmark Runner ====================

class BenchmarkRunner:
    """
    Benchmark运行器
    
    执行完整的benchmark测试流程
    """
    
    def __init__(self,
                 use_mock: bool = True,
                 neo4j_uri: str = None,
                 neo4j_user: str = None,
                 neo4j_password: str = None,
                 openai_api_key: str = None,
                 output_dir: str = "benchmark_results"):
        """
        初始化
        
        Args:
            use_mock: 使用Mock模式
            neo4j_*: Neo4j连接参数
            openai_api_key: OpenAI API密钥
            output_dir: 输出目录
        """
        self.use_mock = use_mock
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        if use_mock:
            self._init_mock_mode()
        else:
            self._init_real_mode(neo4j_uri, neo4j_user, neo4j_password, openai_api_key)
        
        # 评估器
        self.evaluator = CapabilityEvaluator(self.llm_client)
        
        # 结果存储
        self.results = defaultdict(list)
        self.raw_outputs = defaultdict(list)
    
    def _init_mock_mode(self):
        """初始化Mock模式"""
        logger.info("Initializing Mock mode...")
        
        self.db_executor = MockExecutor()
        self.llm_client = MockLLMClient()
        
        # 创建agents
        self.agents = {
            'NeuroXiv-Agent': NeuroXivAgent.create(use_mock=True),
            'Direct LLM': DirectLLMAgent(self.db_executor, self.llm_client),
            'ReAct': ReActAgent(self.db_executor, self.llm_client),
            'Simple RAG': SimpleRAGAgent(self.db_executor, self.llm_client),
        }
    
    def _init_real_mode(self, neo4j_uri, neo4j_user, neo4j_password, openai_api_key):
        """初始化真实模式"""
        logger.info("Initializing Real mode...")
        
        from agent import Neo4jExecutor
        from llm_intelligence import OpenAIClient
        
        self.db_executor = Neo4jExecutor(neo4j_uri, neo4j_user, neo4j_password)
        self.llm_client = OpenAIClient(openai_api_key)
        
        # 创建agents
        self.agents = {
            'NeuroXiv-Agent': NeuroXivAgent.create(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                openai_api_key=openai_api_key
            ),
            'Direct LLM': DirectLLMAgent(self.db_executor, self.llm_client),
            'ReAct': ReActAgent(self.db_executor, self.llm_client),
            'Simple RAG': SimpleRAGAgent(self.db_executor, self.llm_client),
        }
    
    def run_benchmark(self,
                      questions: List[Dict] = None,
                      methods: List[str] = None,
                      verbose: bool = True) -> Dict:
        """
        运行Benchmark
        
        Args:
            questions: 问题列表，默认使用ALL_QUESTIONS
            methods: 要测试的方法列表
            verbose: 是否打印详细信息
        
        Returns:
            完整结果字典
        """
        if questions is None:
            questions = ALL_QUESTIONS
        
        if methods is None:
            methods = list(self.agents.keys())
        
        logger.info(f"Running benchmark: {len(questions)} questions, {len(methods)} methods")
        
        total_runs = len(questions) * len(methods)
        current_run = 0
        
        for question_data in questions:
            qid = question_data['id']
            question = question_data['question']
            
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Question [{qid}]: {question[:60]}...")
            
            for method_name in methods:
                current_run += 1
                
                if verbose:
                    logger.info(f"  [{current_run}/{total_runs}] Running {method_name}...")
                
                try:
                    # 运行Agent
                    agent = self.agents[method_name]
                    start_time = time.time()
                    
                    output = agent.answer(question)
                    
                    # 记录原始输出
                    self.raw_outputs[method_name].append({
                        'question_id': qid,
                        'output': output
                    })
                    
                    # 评估
                    eval_result = self.evaluator.evaluate(
                        question_data=question_data,
                        agent_output=output,
                        method_name=method_name,
                        ground_truth=question_data.get('ground_truth')
                    )
                    
                    self.results[method_name].append(eval_result)
                    
                    if verbose:
                        logger.info(f"    Score: {eval_result.overall_score:.3f} "
                                   f"(T:{eval_result.capability_scores.think:.2f} "
                                   f"P:{eval_result.capability_scores.plan:.2f} "
                                   f"R:{eval_result.capability_scores.reflect:.2f} "
                                   f"A:{eval_result.capability_scores.act:.2f}) "
                                   f"Correct: {eval_result.correctness}")
                
                except Exception as e:
                    logger.error(f"    Error: {e}")
                    # 记录失败结果
                    from evaluator import CapabilityScores
                    fail_result = EvaluationResult(
                        question_id=qid,
                        method=method_name,
                        capability_scores=CapabilityScores(),
                        capability_weighted=0.0,
                        correctness='unanswered',
                        correctness_multiplier=0.1,
                        overall_score=0.0
                    )
                    self.results[method_name].append(fail_result)
        
        # 生成报告
        report = self._generate_report()
        
        # 保存结果
        self._save_results(report)
        
        return report
    
    def _generate_report(self) -> Dict:
        """生成完整报告"""
        logger.info("\nGenerating report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'mock' if self.use_mock else 'real',
            'summary': {},
            'by_method': {},
            'by_level': {},
            'statistical_tests': {},
            'capability_analysis': {},
        }
        
        # 按方法汇总
        for method, results in self.results.items():
            scores = [r.overall_score for r in results]
            capability_scores = {
                'think': [r.capability_scores.think for r in results],
                'plan': [r.capability_scores.plan for r in results],
                'reflect': [r.capability_scores.reflect for r in results],
                'act': [r.capability_scores.act for r in results],
            }
            
            report['by_method'][method] = {
                'n_questions': len(results),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'median_score': float(np.median(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'capability_means': {
                    k: float(np.mean(v)) for k, v in capability_scores.items()
                },
                'correctness_dist': self._count_correctness(results),
            }
        
        # 按难度级别分析
        for level in [1, 2, 3]:
            level_results = {}
            for method, results in self.results.items():
                level_scores = [r.overall_score for r in results 
                               if r.question_id.startswith(f'L{level}')]
                if level_scores:
                    level_results[method] = {
                        'mean': float(np.mean(level_scores)),
                        'std': float(np.std(level_scores)),
                        'n': len(level_scores)
                    }
            report['by_level'][f'level_{level}'] = level_results
        
        # 统计检验
        report['statistical_tests'] = self._run_statistical_tests()
        
        # 能力分析
        report['capability_analysis'] = self._analyze_capabilities()
        
        # 生成总结
        report['summary'] = self._generate_summary(report)
        
        return report
    
    def _count_correctness(self, results: List[EvaluationResult]) -> Dict:
        """统计正确性分布"""
        counts = defaultdict(int)
        for r in results:
            counts[r.correctness] += 1
        return dict(counts)
    
    def _run_statistical_tests(self) -> Dict:
        """运行统计检验"""
        tests = {}
        
        # 获取NeuroXiv-Agent的分数
        neuroxiv_scores = [r.overall_score for r in self.results.get('NeuroXiv-Agent', [])]
        
        if not neuroxiv_scores:
            return tests
        
        # 与每个baseline比较
        for baseline in ['Direct LLM', 'ReAct', 'Simple RAG']:
            baseline_scores = [r.overall_score for r in self.results.get(baseline, [])]
            
            if not baseline_scores:
                continue
            
            # Permutation test
            perm_result = StatisticalValidator.permutation_test(
                neuroxiv_scores, baseline_scores
            )
            
            # Bootstrap CI
            diff = np.array(neuroxiv_scores) - np.mean(baseline_scores)
            ci = StatisticalValidator.bootstrap_ci(diff.tolist())
            
            tests[f'NeuroXiv_vs_{baseline.replace(" ", "_")}'] = {
                'neuroxiv_mean': perm_result['mean1'],
                'baseline_mean': perm_result['mean2'],
                'difference': perm_result['observed_diff'],
                'p_value': perm_result['p_value'],
                'cohens_d': perm_result['cohens_d'],
                'significant': perm_result['significant'],
                'ci_95': ci,
                'interpretation': self._interpret_effect_size(perm_result['cohens_d'])
            }
        
        # FDR校正
        p_values = [t['p_value'] for t in tests.values()]
        if p_values:
            q_values, significant = StatisticalValidator.fdr_correction(p_values)
            for i, (test_name, test_result) in enumerate(tests.items()):
                test_result['q_value'] = q_values[i]
                test_result['significant_fdr'] = significant[i]
        
        return tests
    
    def _interpret_effect_size(self, d: float) -> str:
        """解释效应量"""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_capabilities(self) -> Dict:
        """分析能力差异"""
        analysis = {}
        
        for capability in ['think', 'plan', 'reflect', 'act']:
            cap_data = {}
            
            for method, results in self.results.items():
                scores = [getattr(r.capability_scores, capability) for r in results]
                cap_data[method] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                }
            
            analysis[capability] = cap_data
        
        return analysis
    
    def _generate_summary(self, report: Dict) -> Dict:
        """生成总结"""
        summary = {
            'total_questions': sum(r['n_questions'] for r in report['by_method'].values()) // len(report['by_method']),
            'methods_tested': list(report['by_method'].keys()),
            'best_method': max(report['by_method'].items(), key=lambda x: x[1]['mean_score'])[0],
            'significant_improvements': [],
            'key_findings': [],
        }
        
        # 检查显著改进
        for test_name, test_result in report['statistical_tests'].items():
            if test_result.get('significant_fdr', test_result.get('significant', False)):
                if test_result['difference'] > 0:
                    baseline = test_name.split('_vs_')[1].replace('_', ' ')
                    summary['significant_improvements'].append({
                        'vs': baseline,
                        'improvement': f"{test_result['difference']:.3f}",
                        'p_value': f"{test_result['p_value']:.4f}",
                        'effect_size': test_result['interpretation']
                    })
        
        # 关键发现
        if summary['significant_improvements']:
            summary['key_findings'].append(
                f"NeuroXiv-Agent significantly outperforms {len(summary['significant_improvements'])} baseline(s)"
            )
        
        # 能力优势
        neuroxiv_caps = report['capability_analysis']
        for cap in ['think', 'plan', 'reflect', 'act']:
            neuroxiv_score = neuroxiv_caps[cap].get('NeuroXiv-Agent', {}).get('mean', 0)
            baseline_avg = np.mean([
                neuroxiv_caps[cap].get(m, {}).get('mean', 0) 
                for m in ['Direct LLM', 'ReAct', 'Simple RAG']
            ])
            if neuroxiv_score > baseline_avg * 1.2:  # 20%以上优势
                summary['key_findings'].append(
                    f"Strong advantage in {cap.upper()} capability: {neuroxiv_score:.3f} vs baseline avg {baseline_avg:.3f}"
                )
        
        return summary
    
    def _save_results(self, report: Dict):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整报告
        report_path = os.path.join(self.output_dir, f"report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to: {report_path}")
        
        # 保存详细结果
        details_path = os.path.join(self.output_dir, f"details_{timestamp}.json")
        details = {
            method: [r.to_dict() for r in results]
            for method, results in self.results.items()
        }
        with open(details_path, 'w') as f:
            json.dump(details, f, indent=2, default=str)
        logger.info(f"Details saved to: {details_path}")
        
        # 生成文本报告
        text_report = self._generate_text_report(report)
        text_path = os.path.join(self.output_dir, f"report_{timestamp}.txt")
        with open(text_path, 'w') as f:
            f.write(text_report)
        logger.info(f"Text report saved to: {text_path}")
    
    def _generate_text_report(self, report: Dict) -> str:
        """生成文本报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("NeuroXiv-Agent Benchmark Report")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Mode: {report['mode']}")
        lines.append(f"Total Questions: {report['summary']['total_questions']}")
        lines.append("")
        
        # 方法比较
        lines.append("-" * 70)
        lines.append("METHOD COMPARISON")
        lines.append("-" * 70)
        lines.append(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Median':>10}")
        lines.append("-" * 50)
        
        for method, stats in report['by_method'].items():
            lines.append(f"{method:<20} {stats['mean_score']:>10.4f} {stats['std_score']:>10.4f} {stats['median_score']:>10.4f}")
        
        # 能力分析
        lines.append("")
        lines.append("-" * 70)
        lines.append("CAPABILITY ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"{'Method':<20} {'Think':>10} {'Plan':>10} {'Reflect':>10} {'Act':>10}")
        lines.append("-" * 60)
        
        for method in report['by_method'].keys():
            caps = report['capability_analysis']
            t = caps['think'].get(method, {}).get('mean', 0)
            p = caps['plan'].get(method, {}).get('mean', 0)
            r = caps['reflect'].get(method, {}).get('mean', 0)
            a = caps['act'].get(method, {}).get('mean', 0)
            lines.append(f"{method:<20} {t:>10.4f} {p:>10.4f} {r:>10.4f} {a:>10.4f}")
        
        # 统计检验
        lines.append("")
        lines.append("-" * 70)
        lines.append("STATISTICAL TESTS")
        lines.append("-" * 70)
        
        for test_name, test_result in report['statistical_tests'].items():
            lines.append(f"\n{test_name}:")
            lines.append(f"  Difference: {test_result['difference']:.4f}")
            lines.append(f"  p-value: {test_result['p_value']:.4f}")
            lines.append(f"  Cohen's d: {test_result['cohens_d']:.4f} ({test_result['interpretation']})")
            lines.append(f"  Significant (FDR): {test_result.get('significant_fdr', test_result['significant'])}")
            lines.append(f"  95% CI: [{test_result['ci_95'][0]:.4f}, {test_result['ci_95'][1]:.4f}]")
        
        # 关键发现
        lines.append("")
        lines.append("-" * 70)
        lines.append("KEY FINDINGS")
        lines.append("-" * 70)
        
        for finding in report['summary']['key_findings']:
            lines.append(f"  ★ {finding}")
        
        if report['summary']['significant_improvements']:
            lines.append("\nSignificant Improvements:")
            for imp in report['summary']['significant_improvements']:
                lines.append(f"  • vs {imp['vs']}: +{imp['improvement']} (p={imp['p_value']}, effect={imp['effect_size']})")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def close(self):
        """清理资源"""
        for agent in self.agents.values():
            if hasattr(agent, 'close'):
                agent.close()


# ==================== Visualization ====================

def generate_visualization(report: Dict, output_dir: str):
    """生成可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 方法比较柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(report['by_method'].keys())
    means = [report['by_method'][m]['mean_score'] for m in methods]
    stds = [report['by_method'][m]['std_score'] for m in methods]
    
    colors = ['#2ecc71' if m == 'NeuroXiv-Agent' else '#3498db' for m in methods]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Overall Score')
    ax.set_title('Method Comparison')
    ax.set_ylim(0, 1)
    
    # 添加显著性标记
    for test_name, test_result in report['statistical_tests'].items():
        if test_result.get('significant_fdr', test_result.get('significant', False)):
            baseline = test_name.split('_vs_')[1].replace('_', ' ')
            if baseline in methods:
                idx = methods.index(baseline)
                ax.annotate('*', xy=(idx, means[idx] + stds[idx] + 0.05),
                           ha='center', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{timestamp}.png'), dpi=150)
    plt.close()
    
    # 2. 能力雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    capabilities = ['Think', 'Plan', 'Reflect', 'Act']
    angles = np.linspace(0, 2 * np.pi, len(capabilities), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for method in methods:
        caps = report['capability_analysis']
        values = [
            caps['think'].get(method, {}).get('mean', 0),
            caps['plan'].get(method, {}).get('mean', 0),
            caps['reflect'].get(method, {}).get('mean', 0),
            caps['act'].get(method, {}).get('mean', 0),
        ]
        values += values[:1]
        
        color = '#2ecc71' if method == 'NeuroXiv-Agent' else None
        linewidth = 3 if method == 'NeuroXiv-Agent' else 1
        ax.plot(angles, values, linewidth=linewidth, label=method, color=color)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(capabilities)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Capability Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'capabilities_{timestamp}.png'), dpi=150)
    plt.close()
    
    # 3. 难度级别分析
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, level in enumerate([1, 2, 3]):
        ax = axes[i]
        level_data = report['by_level'].get(f'level_{level}', {})
        
        methods_with_data = [m for m in methods if m in level_data]
        level_means = [level_data[m]['mean'] for m in methods_with_data]
        
        colors = ['#2ecc71' if m == 'NeuroXiv-Agent' else '#3498db' for m in methods_with_data]
        ax.bar(range(len(methods_with_data)), level_means, color=colors, alpha=0.8)
        ax.set_xticks(range(len(methods_with_data)))
        ax.set_xticklabels(methods_with_data, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title(f'Level {level} ({"Simple" if level==1 else "Medium" if level==2 else "Complex"})')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'levels_{timestamp}.png'), dpi=150)
    plt.close()
    
    logger.info(f"Visualizations saved to: {output_dir}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='NeuroXiv-Agent Benchmark Suite')
    
    parser.add_argument('--mode', choices=['mock', 'full'], default='mock',
                       help='Running mode: mock (simulated) or full (real DB/API)')
    parser.add_argument('--questions', type=int, default=None,
                       help='Number of questions (default: all)')
    parser.add_argument('--level', type=int, choices=[1, 2, 3], default=None,
                       help='Only run specific difficulty level')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory')
    parser.add_argument('--neo4j-uri', type=str, default=None,
                       help='Neo4j URI')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--neo4j-password', type=str, default=None,
                       help='Neo4j password')
    parser.add_argument('--openai-key', type=str, default=None,
                       help='OpenAI API key')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # 准备问题
    if args.level:
        questions = get_questions_by_level(args.level)
    else:
        questions = ALL_QUESTIONS
    
    if args.questions:
        questions = questions[:args.questions]
    
    logger.info(f"Selected {len(questions)} questions")
    
    # 创建Runner
    runner = BenchmarkRunner(
        use_mock=(args.mode == 'mock'),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        openai_api_key=args.openai_key or os.environ.get('OPENAI_API_KEY'),
        output_dir=args.output
    )
    
    try:
        # 运行Benchmark
        report = runner.run_benchmark(
            questions=questions,
            verbose=args.verbose
        )
        
        # 打印总结
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"\nBest Method: {report['summary']['best_method']}")
        print(f"\nKey Findings:")
        for finding in report['summary']['key_findings']:
            print(f"  ★ {finding}")
        
        if report['summary']['significant_improvements']:
            print(f"\n✓ NeuroXiv-Agent shows SIGNIFICANT improvements over baselines!")
        
        # 生成可视化
        if args.visualize:
            generate_visualization(report, args.output)
        
    finally:
        runner.close()


if __name__ == '__main__':
    main()