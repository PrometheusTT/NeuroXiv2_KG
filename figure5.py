"""
Brain Region Morphology-Molecular Composition Mismatch Analyzer (Optimized Version)
- Prioritizes regions with sufficient data
- Features diverse, clear visualizations with proper spacing
- Enhanced case selection with meaningful differences
- Improved layout with no text overlapping visuals

Author: wangmajortom
Date: 2025-08-27
"""

import os
import json
import random
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import spearmanr, variation
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Set style for high quality visualizations with proper spacing
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (14, 12)  # Larger default size
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.autolayout'] = True

# Define important brain regions to focus on
IMPORTANT_REGIONS = [
    # Cortical regions
    'V1', 'V3', 'SSp', 'SSs', 'MOs', 'MOp', 'ACAd', 'ACAv', 'ILA',
    'RSP', 'PTLp', 'VISp', 'VISl', 'VISpm', 'VISam', 'TEa', 'PERI',
    # Basal ganglia
    'CP', 'ACB', 'SNc', 'VTA', 'SNr', 'GPe', 'GPi',
    # Thalamic nuclei
    'LGd', 'LP', 'VPM', 'PVH', 'PVT', 'LT', 'MD', 'MG', 'RT', 'POL',
    # Midbrain regions
    'SC', 'IC', 'PAG', 'SCs', 'SCm', 'RN', 'MRN',
    # Limbic system
    'HIP', 'CA1', 'CA3', 'DG', 'ENT', 'AMY', 'BLA', 'CeA', 'BMA'
]

# Define morphological feature names for better labeling
MORPH_FEATURE_NAMES = [
    'Axonal Bifurcation Angle',
    'Axonal Branches',
    'Axonal Length',
    'Axonal Max Branch Order',
    'Dendritic Bifurcation Angle',
    'Dendritic Branches',
    'Dendritic Length',
    'Dendritic Max Branch Order'
]

# Define color palettes for consistent visualization
COLOR_PALETTE = {
    'region1': '#4DBBD5',  # Blue
    'region2': '#E64B35',  # Red
    'mismatch': '#00A087',  # Teal
    'background': '#F0F0F0',
    'highlight': '#FEC44F',  # Yellow/Orange
    'accent': '#7E6148',    # Brown
    'gradient_start': '#EFF3FF',
    'gradient_end': '#3182BD'
}


class MismatchAnalyzer:
    """Enhanced analyzer for brain region morphology-molecular composition mismatch"""

    def __init__(self, morph_vectors, subclass_vectors, region_info, min_data_percentile=30):
        """
        Initialize the enhanced mismatch analyzer

        Parameters:
            morph_vectors: DataFrame of morphological feature vectors
            subclass_vectors: DataFrame of subclass composition vectors
            region_info: DataFrame with region metadata (acronyms, names)
            min_data_percentile: Percentile threshold for data sufficiency (default: 30)
        """
        self.morph_vectors = morph_vectors
        self.subclass_vectors = subclass_vectors
        self.region_info = region_info
        self.region_ids = list(morph_vectors.index)

        # Generate acronym to name mapping
        self.acronym_to_name = dict(zip(
            region_info['acronym'],
            region_info['name'] if 'name' in region_info.columns else region_info['acronym']
        ))

        # Calculate data completeness and variation for regions
        self._calculate_data_quality(min_data_percentile)

        # Filter for important regions with sufficient data
        self.important_regions = [r for r in self.region_ids
                                 if r in self.data_rich_regions and
                                 self.region_info.loc[r, 'acronym'] in IMPORTANT_REGIONS]

        # If we don't have many important regions, just use the top data-rich regions
        if len(self.important_regions) < 15:
            print("Found few important regions with sufficient data. Using top data-rich regions instead.")
            self.important_regions = self.data_rich_regions[:min(30, len(self.data_rich_regions))]

        print(f"Analyzing {len(self.region_ids)} regions in total")
        print(f"Found {len(self.data_rich_regions)} regions with sufficient data")
        print(f"Selected {len(self.important_regions)} important regions for focused analysis")

        # Calculate distance matrices
        self.compute_distance_matrices()

        # Calculate mismatch metrics
        self.compute_mismatch_metrics()

    def _calculate_data_quality(self, min_data_percentile):
        """Calculate data quality metrics for all regions"""
        # Assess data completeness for morphological vectors
        morph_completeness = self.morph_vectors.notna().sum(axis=1) / self.morph_vectors.shape[1]

        # Calculate feature variation (to avoid regions with all zeros or constant values)
        morph_variation = self.morph_vectors.apply(lambda x: variation(x.fillna(0)), axis=1)
        morph_variation = morph_variation.fillna(0)  # Handle any NaN results

        # Assess data quality for subclass vectors
        # For subclass vectors, we consider the number of non-zero values as a measure of completeness
        subclass_nonzero = (self.subclass_vectors != 0).sum(axis=1)
        subclass_percentiles = subclass_nonzero.rank(pct=True) * 100

        # Calculate variation in subclass data
        subclass_variation = self.subclass_vectors.apply(lambda x: variation(x), axis=1)
        subclass_variation = subclass_variation.fillna(0)

        # Combine all scores for a comprehensive quality metric
        # Weight variation more to ensure we select regions with meaningful differences
        quality_score = (
            morph_completeness * 0.3 +
            morph_variation * 0.2 +
            (subclass_percentiles / 100) * 0.3 +
            subclass_variation * 0.2
        )

        # Add quality info to region_info
        self.region_info['morph_completeness'] = morph_completeness
        self.region_info['morph_variation'] = morph_variation
        self.region_info['subclass_percentile'] = subclass_percentiles
        self.region_info['subclass_variation'] = subclass_variation
        self.region_info['data_quality'] = quality_score

        # Filter regions based on quality score
        min_quality = np.percentile(quality_score, min_data_percentile)
        self.data_rich_regions = list(quality_score[quality_score >= min_quality].index)

        print(f"Data quality threshold: {min_quality:.3f}")
        print(f"Identified {len(self.data_rich_regions)} regions with good data quality")

    def compute_distance_matrices(self):
        """Compute morphological and molecular distance matrices"""
        # Calculate morphological distance matrix (Euclidean distance)
        morph_dist = pdist(self.morph_vectors.values, metric='euclidean')
        self.morph_dist_matrix = pd.DataFrame(
            squareform(morph_dist),
            index=self.region_ids,
            columns=self.region_ids
        )

        # Calculate molecular distance matrix (Cosine distance)
        subclass_dist = pdist(self.subclass_vectors.values, metric='cosine')
        self.subclass_dist_matrix = pd.DataFrame(
            squareform(subclass_dist),
            index=self.region_ids,
            columns=self.region_ids
        )

        # Normalize distance matrices for fair comparison
        self.norm_morph_dist = self.morph_dist_matrix / self.morph_dist_matrix.values.max()
        self.norm_subclass_dist = self.subclass_dist_matrix / self.subclass_dist_matrix.values.max()

        print("Distance matrices computed")
        print(f"Morphological distance range: [{self.norm_morph_dist.values.min():.3f}, {self.norm_morph_dist.values.max():.3f}]")
        print(f"Molecular distance range: [{self.norm_subclass_dist.values.min():.3f}, {self.norm_subclass_dist.values.max():.3f}]")

    def compute_mismatch_metrics(self):
        """Compute mismatch metrics between regions"""
        # 1. Calculate absolute difference matrix (mismatch between the two distances)
        self.mismatch_matrix = np.abs(self.norm_morph_dist - self.norm_subclass_dist)

        # 2. Calculate average mismatch index for each region
        self.region_mismatch_index = {}
        for region_id in self.region_ids:
            other_regions = [r for r in self.region_ids if r != region_id]
            mismatch_values = [self.mismatch_matrix.loc[region_id, r] for r in other_regions]
            self.region_mismatch_index[region_id] = np.mean(mismatch_values)

        # 3. Identify significant mismatched pairs among data-rich regions
        self.significant_pairs = []
        for i, r1 in enumerate(self.data_rich_regions):
            for j in range(i+1, len(self.data_rich_regions)):
                r2 = self.data_rich_regions[j]
                morph_dist = self.norm_morph_dist.loc[r1, r2]
                subclass_dist = self.norm_subclass_dist.loc[r1, r2]
                mismatch = self.mismatch_matrix.loc[r1, r2]

                # Get variation data for both regions to ensure meaningful differences
                r1_morph_var = self.region_info.loc[r1, 'morph_variation']
                r2_morph_var = self.region_info.loc[r2, 'morph_variation']
                r1_subclass_var = self.region_info.loc[r1, 'subclass_variation']
                r2_subclass_var = self.region_info.loc[r2, 'subclass_variation']

                # Calculate a feature difference score to ensure visually distinct features
                r1_features = self.morph_vectors.loc[r1]
                r2_features = self.morph_vectors.loc[r2]
                feature_diff = np.mean(np.abs(r1_features - r2_features))

                # Only include pairs with significant mismatch AND good variation in features
                # This ensures we don't get cases where features are all zeros or very similar
                if (mismatch > 0.3 and  # Significant mismatch
                    min(r1_morph_var, r2_morph_var) > 0.1 and  # Good morphological variation
                    min(r1_subclass_var, r2_subclass_var) > 0.1 and  # Good subclass variation
                    feature_diff > 0.2):  # Features are visually distinct

                    self.significant_pairs.append({
                        'region1_id': r1,
                        'region2_id': r2,
                        'region1_acronym': self.region_info.loc[r1, 'acronym'],
                        'region2_acronym': self.region_info.loc[r2, 'acronym'],
                        'morph_distance': morph_dist,
                        'subclass_distance': subclass_dist,
                        'mismatch': mismatch,
                        'type': 'morph_similar' if morph_dist < subclass_dist else 'subclass_similar',
                        'data_quality': (self.region_info.loc[r1, 'data_quality'] +
                                      self.region_info.loc[r2, 'data_quality']) / 2,
                        'feature_diff': feature_diff
                    })

        # Sort pairs by mismatch magnitude and data quality
        self.significant_pairs = sorted(self.significant_pairs,
                                      key=lambda x: (x['mismatch'], x['data_quality'], x['feature_diff']),
                                      reverse=True)

        print(f"Found {len(self.significant_pairs)} significantly mismatched region pairs")
        if self.significant_pairs:
            print(f"Top mismatch: {self.significant_pairs[0]['region1_acronym']}-{self.significant_pairs[0]['region2_acronym']} "
                f"(mismatch={self.significant_pairs[0]['mismatch']:.3f})")

    def plot_global_relationship(self):
        """Plot the global relationship between morphological and molecular distances"""
        # Extract distances for data-rich region pairs
        distances = []
        for i, r1 in enumerate(self.data_rich_regions):
            for j in range(i+1, len(self.data_rich_regions)):
                r2 = self.data_rich_regions[j]
                morph_dist = self.norm_morph_dist.loc[r1, r2]
                subclass_dist = self.norm_subclass_dist.loc[r1, r2]
                is_important = r1 in self.important_regions and r2 in self.important_regions
                r1_acronym = self.region_info.loc[r1, 'acronym']
                r2_acronym = self.region_info.loc[r2, 'acronym']
                pair_label = f"{r1_acronym}-{r2_acronym}"

                distances.append({
                    'Morphological Distance': morph_dist,
                    'Molecular Distance': subclass_dist,
                    'Is Important': is_important,
                    'Pair': pair_label,
                    'Mismatch': abs(morph_dist - subclass_dist)
                })

        # Convert to DataFrame
        dist_df = pd.DataFrame(distances)

        # Calculate correlation
        corr, p_value = spearmanr(dist_df['Morphological Distance'], dist_df['Molecular Distance'])

        # Create large figure with ample spacing
        fig = plt.figure(figsize=(16, 14), facecolor='white')

        # Create a grid layout with reserved space for legend and text
        gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[4, 1])

        # Main scatter plot in top left
        ax_main = fig.add_subplot(gs[0, 0])

        # Create a custom colormap for the mismatch gradient
        mismatch_cmap = LinearSegmentedColormap.from_list('mismatch_cmap',
                                                        [COLOR_PALETTE['gradient_start'],
                                                         COLOR_PALETTE['gradient_end']])

        # Create scatter plot with enhanced styling
        scatter = sns.scatterplot(
            data=dist_df,
            x='Morphological Distance',
            y='Molecular Distance',
            hue='Mismatch',
            palette=mismatch_cmap,
            size='Is Important',
            sizes=(40, 200),
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5,
            ax=ax_main,
            zorder=2
        )

        # Remove the automatically generated legend (we'll create a custom one)
        ax_main.get_legend().remove()

        # Add diagonal line
        max_val = max(dist_df['Morphological Distance'].max(), dist_df['Molecular Distance'].max())
        ax_main.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, linewidth=1.5, label='Equal Distance', zorder=1)

        # Add regression line with confidence interval
        sns.regplot(
            data=dist_df,
            x='Morphological Distance',
            y='Molecular Distance',
            scatter=False,
            color=COLOR_PALETTE['region1'],
            line_kws={'linewidth': 2, 'zorder': 1},
            ci=95,
            ax=ax_main
        )

        # Label top mismatch pairs without overlapping the plot
        top_mismatch = dist_df.nlargest(8, 'Mismatch')

        # Use adjustText if available for better label placement
        try:
            from adjustText import adjust_text
            texts = []
            for _, row in top_mismatch.iterrows():
                texts.append(ax_main.text(
                    row['Morphological Distance'],
                    row['Molecular Distance'],
                    row['Pair'],
                    fontsize=10,
                    fontweight='bold'
                ))
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['accent']))
        except ImportError:
            # Fallback method with manual offsets
            offsets = [(10, 10), (-30, 10), (10, -30), (-30, -30),
                      (30, 5), (-5, 30), (30, -5), (-5, -30)]
            for i, (_, row) in enumerate(top_mismatch.iterrows()):
                offset = offsets[i % len(offsets)]
                ax_main.annotate(
                    row['Pair'],
                    (row['Morphological Distance'], row['Molecular Distance']),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='k',
                    bbox=dict(boxstyle='round,pad=0.3', fc=COLOR_PALETTE['highlight'], ec='none', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color=COLOR_PALETTE['accent'])
                )

        # Add labels and title with enhanced styling
        ax_main.set_xlabel('Normalized Morphological Distance', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('Normalized Molecular Distance', fontsize=14, fontweight='bold')

        title_text = f'Structure-Function Relationship in Brain Regions\nSpearman Correlation: {corr:.3f}'
        title_text += f" (p={p_value:.3e})" if p_value < 0.05 else " (not significant)"

        ax_main.set_title(title_text, fontsize=16, fontweight='bold', pad=20)

        # Add grid with proper z-order
        ax_main.grid(True, linestyle='--', alpha=0.3, zorder=0)

        # Create legend in top-right panel
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_legend.axis('off')

        # Create custom legend elements
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        legend_elements = []

        # Mismatch colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        ax_legend.imshow(gradient, aspect='auto', cmap=mismatch_cmap,
                      extent=[0, 0.8, 0.7, 0.8])
        ax_legend.text(0, 0.65, "Mismatch Index", fontsize=12, fontweight='bold')
        ax_legend.text(0, 0.6, "Low", fontsize=10)
        ax_legend.text(0.7, 0.6, "High", fontsize=10)

        # Size legend
        ax_legend.scatter([0.2, 0.6], [0.5, 0.5], s=[40, 200],
                       color='gray', alpha=0.8, edgecolor='w')
        ax_legend.text(0, 0.45, "Region Importance", fontsize=12, fontweight='bold')
        ax_legend.text(0.2, 0.4, "Standard", fontsize=10)
        ax_legend.text(0.6, 0.4, "Important", fontsize=10)

        # Line legend
        ax_legend.plot([0.1, 0.4], [0.3, 0.3], 'k--', linewidth=1.5)
        ax_legend.text(0.5, 0.3, "Equal Distance", fontsize=10)

        ax_legend.plot([0.1, 0.4], [0.2, 0.2], color=COLOR_PALETTE['region1'], linewidth=2)
        ax_legend.text(0.5, 0.2, "Regression Line", fontsize=10)

        # Set legend boundaries
        ax_legend.set_xlim(-0.1, 1.0)
        ax_legend.set_ylim(0, 1.0)

        # Add explanation boxes to bottom panel
        ax_explanation = fig.add_subplot(gs[1, :])
        ax_explanation.axis('off')

        # Create three explanation boxes side by side
        morph_similar_box = (
            "Morphology Similar, Molecular Different\n"
            "────────────────────────────\n"
            "Regions in this quadrant have similar structural\n"
            "organization but distinct molecular compositions.\n"
            "This suggests divergent functional specialization\n"
            "despite similar morphological constraints."
        )

        mol_similar_box = (
            "Morphology Different, Molecular Similar\n"
            "────────────────────────────\n"
            "Regions in this quadrant have different structural\n"
            "organization but similar molecular compositions.\n"
            "This suggests convergent functional properties\n"
            "despite different morphological adaptations."
        )

        general_box = (
            "Analysis Summary\n"
            "────────────────\n"
            f"• Data-rich regions: {len(self.data_rich_regions)}\n"
            f"• Total region pairs: {len(distances)}\n"
            f"• Correlation strength: {abs(corr):.2f} ({'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'})\n"
            f"• Mean mismatch: {dist_df['Mismatch'].mean():.3f}\n"
            f"• Max mismatch: {dist_df['Mismatch'].max():.3f}"
        )

        # Position the text boxes
        ax_explanation.text(0.01, 0.95, morph_similar_box, fontsize=11, va='top',
                         bbox=dict(boxstyle='round', fc='white', ec=COLOR_PALETTE['region2'], lw=2))

        ax_explanation.text(0.35, 0.95, general_box, fontsize=11, va='top',
                         bbox=dict(boxstyle='round', fc='white', ec=COLOR_PALETTE['accent'], lw=2))

        ax_explanation.text(0.68, 0.95, mol_similar_box, fontsize=11, va='top',
                         bbox=dict(boxstyle='round', fc='white', ec=COLOR_PALETTE['region1'], lw=2))

        return fig

    def plot_mismatch_heatmap(self):
        """Plot an enhanced heatmap of mismatches between important brain regions"""
        # Filter important regions
        important_acronyms = [self.region_info.loc[r, 'acronym'] for r in self.important_regions]

        if len(self.important_regions) < 5:
            print("Too few important regions to create heatmap")
            return None

        # Extract mismatch submatrix for important regions
        mismatch_submatrix = self.mismatch_matrix.loc[self.important_regions, self.important_regions]

        # Create figure with separate spaces for heatmap and legend/explanation
        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

        # Main heatmap area
        ax_heatmap = fig.add_subplot(gs[0, 0])

        # Define custom colormap
        cmap = LinearSegmentedColormap.from_list('mismatch_cmap',
                                                [COLOR_PALETTE['gradient_start'],
                                                 '#FFF7BC',
                                                 '#FEC44F',
                                                 '#EC7014',
                                                 '#CC4C02',
                                                 '#8C2D04'])

        # Create mask for upper triangle to show only lower triangle
        mask = np.triu(np.ones_like(mismatch_submatrix, dtype=bool), k=0)

        # Create heatmap with enhanced styling
        heatmap = sns.heatmap(
            mismatch_submatrix,
            cmap=cmap,
            mask=mask,  # Apply mask to show only lower triangle
            xticklabels=important_acronyms,
            yticklabels=important_acronyms,
            vmin=0,
            vmax=np.percentile(self.mismatch_matrix.values, 95),  # Use 95th percentile as max value
            square=True,
            cbar_kws={'label': 'Mismatch Index',
                     'orientation': 'horizontal',
                     'shrink': 0.8,
                     'aspect': 40,
                     'pad': 0.12}
        )

        # Set labels and title with enhanced styling
        plt.title('Brain Region Morphology-Molecular Mismatch Matrix',
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Brain Region', fontsize=14, fontweight='bold')
        plt.ylabel('Brain Region', fontsize=14, fontweight='bold')

        # Adjust tick labels
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        # Highlight high mismatch pairs
        threshold = np.percentile(self.mismatch_matrix.values, 90)

        # Find coordinates of high mismatch pairs
        high_mismatch_coords = []
        for i in range(len(self.important_regions)):
            for j in range(len(self.important_regions)):
                if i > j and mismatch_submatrix.iloc[i, j] > threshold:
                    high_mismatch_coords.append((j, i))

        # Add labels for high mismatch pairs
        for j, i in high_mismatch_coords:
            value = mismatch_submatrix.iloc[i, j]
            color = 'black' if value < 0.6 else 'white'  # Adjust text color based on background
            ax_heatmap.text(j + 0.5, i + 0.5, f"{value:.2f}",
                        ha='center', va='center',
                        color=color, fontweight='bold')

        # Add explanation panel on right side
        ax_explain = fig.add_subplot(gs[0, 1])
        ax_explain.axis('off')

        # Title for explanation panel
        ax_explain.text(0.5, 0.98, "Matrix Interpretation Guide",
                     fontsize=14, fontweight='bold', ha='center', va='top')

        # Main explanation
        explanation_text = (
            "This matrix shows the mismatch between morphological\n"
            "and molecular distances for pairs of brain regions.\n\n"
            "Higher values (darker colors) indicate greater\n"
            "discrepancies between structure and function.\n\n"
            f"The matrix displays {len(self.important_regions)} important regions\n"
            f"with sufficient data quality.\n\n"
            f"Values above the {threshold:.2f} threshold are labeled.\n\n"
            "Only the lower triangle is shown to avoid redundancy."
        )

        ax_explain.text(0.5, 0.85, explanation_text,
                     fontsize=12, ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                             edgecolor=COLOR_PALETTE['accent']))

        # Example interpretation
        example_text = (
            "Example Interpretation:\n\n"
            "If regions A and B have a high mismatch value,\n"
            "this means their morphological distance and\n"
            "molecular distance are very different from each other.\n\n"
            "For instance, they might be morphologically similar\n"
            "but molecularly distinct, or vice versa.\n\n"
            "These mismatches represent interesting cases\n"
            "where structure and function have potentially\n"
            "evolved independently."
        )

        ax_explain.text(0.5, 0.4, example_text,
                     fontsize=11, ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='#FFF7BC', alpha=0.8,
                             edgecolor='#EC7014'))

        return fig

    def plot_region_mismatch_profile(self, top_n=15):
        """Plot enhanced mismatch profiles of regions with rich visual elements"""
        # Get mismatch indices for data-rich regions only
        data_rich_mismatch = {r: self.region_mismatch_index[r] for r in self.data_rich_regions}

        # Sort regions by mismatch index
        sorted_regions = sorted(data_rich_mismatch.items(), key=lambda x: x[1], reverse=True)
        top_regions = sorted_regions[:top_n]

        # Extract plotting data
        region_ids = [r[0] for r in top_regions]
        region_acronyms = [self.region_info.loc[r_id, 'acronym'] for r_id in region_ids]
        mismatch_values = [r[1] for r in top_regions]

        # Calculate morphological and molecular complexity
        morph_complexity = [np.linalg.norm(self.morph_vectors.loc[r_id]) for r_id in region_ids]
        molecular_complexity = [np.linalg.norm(self.subclass_vectors.loc[r_id]) for r_id in region_ids]

        # Normalize complexity scores
        morph_complexity = [(x - min(morph_complexity)) / (max(morph_complexity) - min(morph_complexity) + 1e-10)
                           for x in morph_complexity]
        molecular_complexity = [(x - min(molecular_complexity)) / (max(molecular_complexity) - min(molecular_complexity) + 1e-10)
                              for x in molecular_complexity]

        # Create DataFrame
        plot_data = pd.DataFrame({
            'Region': region_acronyms,
            'Mismatch Index': mismatch_values,
            'Morphological Complexity': morph_complexity,
            'Molecular Complexity': molecular_complexity
        })

        # Create more visually appealing figure with well-separated subplots
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3)

        # 1. Main bar chart with gradient colors
        ax_main = fig.add_subplot(gs[0, 0])

        # Create horizontal bar chart with gradient colors
        regions = plot_data['Region']
        y_pos = np.arange(len(regions))

        # Create color gradients based on mismatch values
        colors = plt.cm.YlOrRd(plot_data['Mismatch Index'] / max(plot_data['Mismatch Index']))

        # Plot horizontal bars
        bars = ax_main.barh(y_pos, plot_data['Mismatch Index'], color=colors,
                          edgecolor='black', linewidth=0.5, alpha=0.85)

        # Add complexity markers - placed next to the bars, not on top
        marker_offset = 0.02
        for i, (morph, mol) in enumerate(zip(plot_data['Morphological Complexity'],
                                           plot_data['Molecular Complexity'])):
            # Calculate position after the bar
            bar_end = plot_data['Mismatch Index'][i]

            # Add markers for complexity after the bars
            ax_main.plot(bar_end + marker_offset, i, 'o',
                       markersize=morph * 15 + 5, color=COLOR_PALETTE['region1'],
                       alpha=0.7, label='Morph' if i == 0 else "")

            ax_main.plot(bar_end + marker_offset*3, i, '*',
                       markersize=mol * 15 + 5, color=COLOR_PALETTE['region2'],
                       alpha=0.7, label='Mol' if i == 0 else "")

        # Improve formatting and styling
        ax_main.set_yticks(y_pos)
        ax_main.set_yticklabels(regions, fontsize=12)
        ax_main.set_xlabel('Mismatch Index', fontsize=14, fontweight='bold')
        ax_main.set_title('Top Regions with Structure-Function Mismatch',
                        fontsize=16, fontweight='bold', pad=20)

        # Add value labels
        for i, v in enumerate(plot_data['Mismatch Index']):
            ax_main.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10)

        # Extend x-axis to make room for markers
        max_mismatch = max(plot_data['Mismatch Index'])
        ax_main.set_xlim(0, max_mismatch * 1.2)

        # Add grid lines
        ax_main.grid(True, axis='x', linestyle='--', alpha=0.3)

        # Add custom legend for complexity markers - placed at top of chart for visibility
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_PALETTE['region1'],
                 label='Morphological Complexity', markersize=10),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_PALETTE['region2'],
                 label='Molecular Complexity', markersize=12)
        ]
        ax_main.legend(handles=legend_elements, loc='upper right', frameon=True,
                     bbox_to_anchor=(1.0, 1.05))

        # 2. Scatter plot comparing morphological vs molecular complexity
        ax_scatter = fig.add_subplot(gs[1, 0])

        scatter = ax_scatter.scatter(
            plot_data['Morphological Complexity'],
            plot_data['Molecular Complexity'],
            c=plot_data['Mismatch Index'],
            cmap='YlOrRd',
            s=100,
            alpha=0.8,
            edgecolor='k',
            linewidth=0.5
        )

        # Add region labels to scatter points with offset to avoid overlap
        for i, region in enumerate(plot_data['Region']):
            ax_scatter.annotate(
                region,
                (plot_data['Morphological Complexity'][i], plot_data['Molecular Complexity'][i]),
                fontsize=9,
                ha='center',
                va='center',
                xytext=(5, 5),
                textcoords='offset points'
            )

        # Add diagonal line
        lims = [0, 1]
        ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='Equal Complexity')

        ax_scatter.set_xlabel('Morphological Complexity', fontsize=12)
        ax_scatter.set_ylabel('Molecular Complexity', fontsize=12)
        ax_scatter.set_title('Complexity Comparison', fontsize=14)
        ax_scatter.grid(True, linestyle='--', alpha=0.3)
        ax_scatter.legend(loc='upper left')

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax_scatter)
        cbar.set_label('Mismatch Index', fontsize=10)

        # 3. Information box with stats and interpretation - right side column
        ax_info = fig.add_subplot(gs[:, 1])
        ax_info.axis('off')

        # Calculate statistics for explanation
        avg_mismatch = np.mean(mismatch_values)
        high_morph_low_mol = sum(1 for i in range(len(morph_complexity))
                              if morph_complexity[i] > 0.7 and molecular_complexity[i] < 0.3)
        low_morph_high_mol = sum(1 for i in range(len(morph_complexity))
                              if morph_complexity[i] < 0.3 and molecular_complexity[i] > 0.7)

        # Prepare section titles with decorative elements
        ax_info.text(0.5, 0.98, "ANALYSIS OF TOP MISMATCHED REGIONS",
                  fontsize=14, fontweight='bold', ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['gradient_end'],
                          edgecolor='none', alpha=0.2))

        # Key statistics section
        stats_text = (
            f"Average mismatch index: {avg_mismatch:.3f}\n\n"
            f"Complexity patterns in top regions:\n"
            f"• {high_morph_low_mol} regions with high morphological\n"
            f"  but low molecular complexity\n"
            f"• {low_morph_high_mol} regions with low morphological\n"
            f"  but high molecular complexity"
        )

        ax_info.text(0.5, 0.85, "KEY STATISTICS",
                  fontsize=12, fontweight='bold', ha='center', va='top')

        ax_info.text(0.5, 0.8, stats_text,
                  ha='center', va='top', fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white',
                          edgecolor=COLOR_PALETTE['accent'],
                          alpha=0.9))

        # Interpretation section
        interpretation_text = (
            "Regions with high mismatch indices represent brain areas\n"
            "where structural features and molecular composition show\n"
            "divergent patterns.\n\n"
            "These mismatches may reflect evolutionary adaptations\n"
            "or specialized functional roles where either structure\n"
            "or molecular composition has been independently shaped\n"
            "by different selection pressures.\n\n"
            "The mismatch index captures how much the morphological\n"
            "distance between regions differs from their molecular\n"
            "distance. Higher values indicate greater discrepancy."
        )

        ax_info.text(0.5, 0.5, "INTERPRETATION",
                  fontsize=12, fontweight='bold', ha='center', va='top')

        ax_info.text(0.5, 0.45, interpretation_text,
                  ha='center', va='top', fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white',
                          edgecolor=COLOR_PALETTE['region2'],
                          alpha=0.9))

        # What makes a good case section
        good_case_text = (
            "The ideal case for detailed study exhibits:\n\n"
            "1. High mismatch index (>0.3)\n"
            "2. Significant variation in both morphological\n"
            "   and molecular features\n"
            "3. Clear visual distinction between features\n"
            "4. Good data completeness for both regions"
        )

        ax_info.text(0.5, 0.18, "WHAT MAKES A GOOD CASE",
                  fontsize=12, fontweight='bold', ha='center', va='top')

        ax_info.text(0.5, 0.12, good_case_text,
                  ha='center', va='top', fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.5',
                          facecolor=COLOR_PALETTE['highlight'],
                          edgecolor='none',
                          alpha=0.5))

        return fig

    def create_case_study_visualization(self, case_type='both', n_cases=3):
        """
        Create enhanced case study visualizations with diverse visual elements

        Parameters:
            case_type: 'morph_similar' (morphologically similar but molecularly different)
                      'subclass_similar' (molecularly similar but morphologically different)
                      'both' (show both types)
            n_cases: Number of cases to show for each type
        """
        # Filter cases by type with preference for data quality and feature difference
        if case_type == 'both':
            morph_similar_cases = sorted([p for p in self.significant_pairs if p['type'] == 'morph_similar'],
                                       key=lambda x: (x['mismatch'], x['data_quality'], x['feature_diff']),
                                       reverse=True)[:n_cases]

            subclass_similar_cases = sorted([p for p in self.significant_pairs if p['type'] == 'subclass_similar'],
                                          key=lambda x: (x['mismatch'], x['data_quality'], x['feature_diff']),
                                          reverse=True)[:n_cases]

            cases = morph_similar_cases + subclass_similar_cases
        else:
            cases = sorted([p for p in self.significant_pairs if p['type'] == case_type],
                         key=lambda x: (x['mismatch'], x['data_quality'], x['feature_diff']),
                         reverse=True)[:n_cases]

        if not cases:
            print("No cases found matching criteria")
            return None

        # Process each case with a dedicated multi-panel figure
        all_figures = []

        for i, case in enumerate(cases):
            r1_id, r2_id = case['region1_id'], case['region2_id']
            r1_acro, r2_acro = case['region1_acronym'], case['region2_acronym']

            # Double check for meaningful differences in features
            r1_features = self.morph_vectors.loc[r1_id].values
            r2_features = self.morph_vectors.loc[r2_id].values
            feature_diff = np.abs(r1_features - r2_features)

            # Skip this case if features are too similar (all differences < 0.1)
            if np.all(feature_diff < 0.1):
                print(f"Skipping case {r1_acro}-{r2_acro} due to insufficient feature differences")
                continue

            # Create an individual figure for each case
            fig = self._create_single_case_visualization(i+1, case)
            all_figures.append(fig)

        return all_figures

    def _create_single_case_visualization(self, case_num, case):
        """Create visualization for a single case study with significantly improved spacing"""
        r1_id, r2_id = case['region1_id'], case['region2_id']
        r1_acro, r2_acro = case['region1_acronym'], case['region2_acronym']

        # Get vector data
        morph_r1 = self.morph_vectors.loc[r1_id].values
        morph_r2 = self.morph_vectors.loc[r2_id].values
        subclass_r1 = self.subclass_vectors.loc[r1_id].values
        subclass_r2 = self.subclass_vectors.loc[r2_id].values

        # Create much larger figure with generous spacing
        fig = plt.figure(figsize=(24, 20), facecolor='white')

        # Use a more spacious grid layout with larger gaps
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2],
                               hspace=0.4, wspace=0.4)

        # 1. Header panel with key metrics and interpretation - spans full width
        ax_header = fig.add_subplot(gs[0, :])
        self._create_header_panel_improved(ax_header, case)

        # 2. Radar chart for morphological features - larger subplot
        ax_radar = fig.add_subplot(gs[1, 0], polar=True)
        self._create_morphology_radar_improved(ax_radar, morph_r1, morph_r2, r1_acro, r2_acro)

        # 3. Molecular composition visualization - larger subplot
        ax_mol = fig.add_subplot(gs[1, 1])
        # Use enhanced method that records and visualizes PCA metrics
        molecular_metrics = self._create_molecular_visualization_improved(
            ax_mol, subclass_r1, subclass_r2, r1_id, r2_id, r1_acro, r2_acro
        )

        # 4. Bottom row - split into two parts
        # Create a subgrid for the bottom row
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :], wspace=0.4)

        # 4a. Left side: Distance heatmaps with explanation
        ax_heatmap_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_bottom[0, 0], wspace=0.3)
        ax_morph_heat = fig.add_subplot(ax_heatmap_grid[0, 0])
        ax_mol_heat = fig.add_subplot(ax_heatmap_grid[0, 1])
        self._create_distance_heatmaps_improved(ax_morph_heat, ax_mol_heat, r1_id, r2_id, r1_acro, r2_acro)

        # 4b. Right side: Combined 3D visualization and metrics chart
        gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_bottom[0, 1], wspace=0.3)

        # 3D visualization
        ax_3d = fig.add_subplot(gs_right[0, 0], projection='3d')
        self._create_3d_feature_visualization_improved(ax_3d, r1_id, r2_id, r1_acro, r2_acro)

        # Metrics chart
        ax_metrics = fig.add_subplot(gs_right[0, 1])
        self._create_metrics_comparison_improved(ax_metrics, case)

        # Add overall title with generous padding
        fig.suptitle(f'Case Study #{case_num}: {r1_acro} vs {r2_acro}',
                     fontsize=24, fontweight='bold', y=0.98)

        return fig

    def _create_header_panel_improved(self, ax, case):
        """Create improved header panel with better spacing"""
        ax.axis('off')

        # Get region names and details
        r1_acro, r2_acro = case['region1_acronym'], case['region2_acronym']
        r1_name = self.acronym_to_name.get(r1_acro, r1_acro)
        r2_name = self.acronym_to_name.get(r2_acro, r2_acro)

        # Determine case type and prepare explanation
        if case['type'] == 'morph_similar':
            case_title = "Similar Structure, Different Function"
            bg_color = '#F0F7FF'  # Light blue
            accent_color = COLOR_PALETTE['region1']
        else:  # subclass_similar
            case_title = "Different Structure, Similar Function"
            bg_color = '#FFF0F0'  # Light red
            accent_color = COLOR_PALETTE['region2']

        # Draw background rectangle
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             facecolor=bg_color, alpha=0.3, zorder=1)
        ax.add_patch(rect)

        # Draw vertical accent bar
        accent = plt.Rectangle((0, 0), 0.01, 1, transform=ax.transAxes,
                               facecolor=accent_color, alpha=1.0, zorder=2)
        ax.add_patch(accent)

        # Add case title - positioned with ample spacing
        ax.text(0.05, 0.65, case_title,
                fontsize=22, fontweight='bold', color='black', transform=ax.transAxes)

        # Add region names - widely spaced for clarity
        ax.text(0.05, 0.35, f"{r1_acro}: {r1_name}",
                fontsize=16, color=COLOR_PALETTE['region1'], transform=ax.transAxes)
        ax.text(0.05, 0.15, f"{r2_acro}: {r2_name}",
                fontsize=16, color=COLOR_PALETTE['region2'], transform=ax.transAxes)

        # Add key metrics - positioned to the right with ample space
        metrics_text = (
            f"Morphological similarity: {1 - case['morph_distance']:.2f}\n"
            f"Molecular similarity: {1 - case['subclass_distance']:.2f}\n"
            f"Mismatch index: {case['mismatch']:.2f}"
        )

        ax.text(0.4, 0.5, metrics_text,
                fontsize=16, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9))

        # Add interpretation - positioned far right
        if case['type'] == 'morph_similar':
            interp_text = (
                "These regions have remarkably similar morphological features\n"
                "but significantly different molecular compositions.\n\n"
                "This suggests that despite similar structural organization,\n"
                "they likely serve distinct functional roles in neural processing."
            )
        else:  # subclass_similar
            interp_text = (
                "These regions have distinct morphological characteristics\n"
                "but surprisingly similar molecular compositions.\n\n"
                "This suggests that despite different structural organizations,\n"
                "they may share similar functional properties or cell types."
            )

        ax.text(0.7, 0.5, interp_text, fontsize=16, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9))

    def _create_morphology_radar_improved(self, ax, morph_r1, morph_r2, r1_acro, r2_acro):
        """Create enhanced radar chart with better spacing"""
        # Normalize vectors for visualization
        max_vals = np.maximum(morph_r1, morph_r2)
        min_vals = np.minimum(morph_r1, morph_r2)

        # Get absolute max for scaling
        abs_max = np.max(np.abs([morph_r1, morph_r2]))

        # Scale to [-1, 1] range
        morph_r1_norm = morph_r1 / (abs_max + 1e-10)
        morph_r2_norm = morph_r2 / (abs_max + 1e-10)

        # Calculate mean values for annotations
        mean_r1 = np.mean(morph_r1_norm)
        mean_r2 = np.mean(morph_r2_norm)

        # Set up radar chart angles
        feature_count = len(morph_r1)
        angles = np.linspace(0, 2 * np.pi, feature_count, endpoint=False).tolist()

        # Make the plot circular by appending the first value to the end
        morph_r1_plot = np.append(morph_r1_norm, morph_r1_norm[0])
        morph_r2_plot = np.append(morph_r2_norm, morph_r2_norm[0])
        angles = np.append(angles, angles[0])

        # Feature labels
        if len(MORPH_FEATURE_NAMES) >= feature_count:
            feature_labels = MORPH_FEATURE_NAMES[:feature_count]
        else:
            feature_labels = [f"Feature {i + 1}" for i in range(feature_count)]

        # Plot radar chart with enhanced styling
        ax.fill(angles, morph_r1_plot, color=COLOR_PALETTE['region1'], alpha=0.25)
        ax.plot(angles, morph_r1_plot, 'o-', linewidth=2, color=COLOR_PALETTE['region1'],
                label=r1_acro, zorder=10)

        ax.fill(angles, morph_r2_plot, color=COLOR_PALETTE['region2'], alpha=0.25)
        ax.plot(angles, morph_r2_plot, 's-', linewidth=2, color=COLOR_PALETTE['region2'],
                label=r2_acro, zorder=10)

        # Add mean value markers - positioned well outside the radar chart
        ax.text(0.05, 1.25, f"{r1_acro} Mean: {mean_r1:.2f}",
                transform=ax.transAxes,
                color=COLOR_PALETTE['region1'],
                fontweight='bold',
                fontsize=14)

        ax.text(0.05, 1.15, f"{r2_acro} Mean: {mean_r2:.2f}",
                transform=ax.transAxes,
                color=COLOR_PALETTE['region2'],
                fontweight='bold',
                fontsize=14)

        # Set up radar chart styling
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set radar chart outer boundary
        ax.set_ylim(-1.1, 1.1)

        # Add feature labels with better spacing
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels, fontsize=12)

        # Add subtle grid and axis lines
        ax.grid(True, color='gray', alpha=0.2, linestyle='--')

        # Add concentric circles with labels for reference
        circles = [-1, -0.5, 0, 0.5, 1]
        for circle in circles:
            ax.add_patch(plt.Circle((0, 0), radius=circle, fill=False,
                                    color='gray', alpha=0.1, zorder=1))
            # Add label except at 0
            if circle != 0:
                ax.text(-0.2, circle, str(circle), verticalalignment='center',
                        fontsize=10, alpha=0.7)

        # Add legend far outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.2), frameon=True,
                  fontsize=14, framealpha=0.9)

        # Add title with generous padding
        ax.set_title('Morphological Features', fontsize=18, fontweight='bold', y=1.3)

        # Add feature explanation far outside the plot
        feature_text = "\n".join([f"{i + 1}. {name}" for i, name in enumerate(feature_labels)])
        ax.text(1.3, 0.5, feature_text, transform=ax.transAxes,
                fontsize=12, va='center')

    def _create_molecular_visualization_improved(self, ax, subclass_r1, subclass_r2, r1_id, r2_id, r1_acro, r2_acro):
        """
        Create spacious molecular composition visualization with PCA analysis
        """
        # Use more vertical space by creating a taller grid with more spacing
        # Reduce from 2 panels to 1 panel with spacious layout
        fig = plt.figure(figsize=(14, 18))  # Create temporary figure to get GridSpec
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.2, 2], hspace=0.35)

        # PCA visualization subplot - top panel
        ax_pca = fig.add_subplot(gs[0, 0])

        # Middle panel for spacing and explanation
        ax_exp = fig.add_subplot(gs[1, 0])
        ax_exp.axis('off')  # Hide axes, just for spacing

        # Subclass bar chart subplot - bottom panel with more space
        ax_bars = fig.add_subplot(gs[2, 0])

        # Calculate full vector similarity (using all dimensions)
        full_similarity = 1 - cosine(subclass_r1, subclass_r2)

        # Prepare data for PCA - select up to 300 dimensions to ensure computational efficiency
        max_dims = min(300, len(subclass_r1))
        r1_sample = subclass_r1[:max_dims]
        r2_sample = subclass_r2[:max_dims]

        # ===== PCA ANALYSIS =====
        # Run PCA on a sample of brain regions to establish feature space
        # Get a sample of regions (including our two regions of interest)
        sample_size = min(30, len(self.region_ids))  # Reduced from 50 to reduce crowding
        sample_regions = random.sample(self.region_ids, sample_size)

        # Make sure our two regions are in the sample
        if r1_id not in sample_regions:
            sample_regions[0] = r1_id
        if r2_id not in sample_regions:
            sample_regions[1] = r2_id

        # Get data for these regions
        sample_data = [self.subclass_vectors.loc[r_id].values[:max_dims] for r_id in sample_regions]
        sample_data = np.vstack(sample_data)

        # Run PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(sample_data)
        variance_explained = pca.explained_variance_ratio_

        # Get coordinates of our two regions
        r1_idx = sample_regions.index(r1_id)
        r2_idx = sample_regions.index(r2_id)
        r1_pca = pca_result[r1_idx]
        r2_pca = pca_result[r2_idx]

        # Calculate feature importance from PCA components
        feature_importance = np.abs(pca.components_[0]) + np.abs(pca.components_[1])

        # Plot the PCA results
        # First plot all sample points
        ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], c='lightgray', alpha=0.5, s=50)

        # Then highlight our two regions with larger markers
        ax_pca.scatter(r1_pca[0], r1_pca[1], c=COLOR_PALETTE['region1'], s=250, label=r1_acro, edgecolor='k',
                       linewidth=1.5)
        ax_pca.scatter(r2_pca[0], r2_pca[1], c=COLOR_PALETTE['region2'], s=250, label=r2_acro, edgecolor='k',
                       linewidth=1.5)

        # Connect the two points
        ax_pca.plot([r1_pca[0], r2_pca[0]], [r1_pca[1], r2_pca[1]], 'k--', alpha=0.7, linewidth=2)

        # Calculate Euclidean distance in PCA space
        pca_distance = np.sqrt(np.sum((r1_pca - r2_pca) ** 2))

        # Add distance annotation with more space
        midpoint_x = (r1_pca[0] + r2_pca[0]) / 2
        midpoint_y = (r1_pca[1] + r2_pca[1]) / 2
        ax_pca.annotate(f"PCA dist: {pca_distance:.2f}",
                        (midpoint_x, midpoint_y),
                        xytext=(15, 15),
                        textcoords='offset points',
                        fontsize=14,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Axis labels with variance explained
        ax_pca.set_xlabel(f'PC1 ({variance_explained[0]:.1%} variance explained)', fontsize=14)
        ax_pca.set_ylabel(f'PC2 ({variance_explained[1]:.1%} variance explained)', fontsize=14)
        ax_pca.set_title(f'Molecular PCA Analysis\nFull Vector Similarity: {full_similarity:.2f}', fontsize=18)

        # Move legend outside the plot for more space
        ax_pca.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=14)

        # Add more grid lines for reference
        ax_pca.grid(alpha=0.3, linestyle='--')

        # Add explanation text in middle panel
        explanation = (
            "PCA visualizes high-dimensional molecular data in 2D. "
            f"Points represent {sample_size} brain regions, with our regions of interest highlighted. "
            "Closer points have more similar molecular compositions."
        )
        ax_exp.text(0.5, 0.5, explanation,
                    ha='center', va='center', fontsize=14,
                    transform=ax_exp.transAxes)

        # ===== SUBCLASS SELECTION & VISUALIZATION =====
        # Determine subclass selection using balanced strategy
        # 1. Calculate mean expression for overall importance
        mean_expression = (np.abs(r1_sample) + np.abs(r2_sample)) / 2

        # 2. Calculate absolute difference for discrimination power
        abs_diff = np.abs(r1_sample - r2_sample)

        # 3. Calculate relative difference (normalized by mean expression)
        # Add small value to avoid division by zero
        relative_diff = abs_diff / (mean_expression + 1e-10)

        # 4. Create combined score balancing:
        #    - PCA importance (global structure)
        #    - Expression level (biological significance)
        #    - Absolute difference (discriminative power)
        combined_score = (
                0.3 * feature_importance / (np.max(feature_importance) + 1e-10) +
                0.3 * mean_expression / (np.max(mean_expression) + 1e-10) +
                0.4 * abs_diff / (np.max(abs_diff) + 1e-10)
        )

        # 5. Filter out subclasses with very low expression in both regions
        expression_threshold = 0.05 * np.max(mean_expression)
        valid_indices = np.where(mean_expression > expression_threshold)[0]

        # 6. Select top subclasses from valid indices
        # Reduced from 8 to 6 for more spacing
        n_subclasses = min(6, len(valid_indices))
        if len(valid_indices) == 0:
            # Fallback if no valid indices
            top_indices = np.argsort(combined_score)[::-1][:n_subclasses]
        else:
            # Get indices with highest combined score among valid indices
            sorted_valid_indices = valid_indices[np.argsort(combined_score[valid_indices])[::-1]]
            top_indices = sorted_valid_indices[:n_subclasses]

        # Extract values for visualization
        r1_values = [r1_sample[i] for i in top_indices]
        r2_values = [r2_sample[i] for i in top_indices]

        # Create labels (use actual subclass names if available)
        if hasattr(self, 'subclass_names') and len(self.subclass_names) >= max_dims:
            labels = [self.subclass_names[i] for i in top_indices]
        else:
            labels = [f"SC{i + 1}" for i in range(len(top_indices))]

        # Calculate differences for sorting and annotation
        diffs = [abs(r1 - r2) for r1, r2 in zip(r1_values, r2_values)]

        # Sort by absolute difference for better visualization
        sorted_indices = np.argsort(diffs)[::-1]

        r1_values = [r1_values[i] for i in sorted_indices]
        r2_values = [r2_values[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        diffs = [diffs[i] for i in sorted_indices]

        # Calculate similarity of just the selected subclasses
        selected_similarity = 1 - cosine(np.array(r1_values), np.array(r2_values))

        # Record metrics for export
        metrics = {
            'region1_id': r1_id,
            'region2_id': r2_id,
            'region1_acronym': r1_acro,
            'region2_acronym': r2_acro,
            'full_vector_similarity': full_similarity,
            'selected_subclass_similarity': selected_similarity,
            'pca_variance_explained': variance_explained.tolist(),
            'pca_distance': float(pca_distance),
            'pca_coordinates': {
                r1_acro: r1_pca.tolist(),
                r2_acro: r2_pca.tolist()
            },
            'pca_top_features': feature_importance.argsort()[::-1][:10].tolist(),
            'pca_top_feature_scores': [float(feature_importance[i]) for i in feature_importance.argsort()[::-1][:10]],
            'selected_subclasses': top_indices.tolist(),
            'timestamp': '2025-08-27 12:58:31',
            'username': 'wangmajortom'
        }

        if not hasattr(self, 'molecular_analysis_metrics'):
            self.molecular_analysis_metrics = []
        self.molecular_analysis_metrics.append(metrics)

        # Create horizontal grouped bar chart with generous spacing
        y_pos = np.arange(len(labels)) * 1.5  # Increased spacing between bars by 50%
        width = 0.5  # Wider bars

        # Plot bars
        ax_bars.barh(y_pos - width / 2, r1_values, width, color=COLOR_PALETTE['region1'], label=r1_acro)
        ax_bars.barh(y_pos + width / 2, r2_values, width, color=COLOR_PALETTE['region2'], label=r2_acro)

        # Add difference annotations - positioned far after all bars
        max_val = max(max(r1_values), max(r2_values))
        annotation_pos = max_val * 1.3  # More space for annotations

        for i, diff in enumerate(diffs):
            ax_bars.text(annotation_pos, y_pos[i], f"Δ = {diff:.2f}",
                         va='center', fontsize=14, color='black')

            # Add percent difference with more space
            mean_val = (abs(r1_values[i]) + abs(r2_values[i])) / 2
            if mean_val > 0:  # Avoid division by zero
                percent_diff = (diff / mean_val) * 100
                ax_bars.text(annotation_pos, y_pos[i] + 0.3, f"({percent_diff:.1f}%)",
                             va='center', fontsize=12, color='gray')

        # Add styling with larger text
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(labels, fontsize=14)
        ax_bars.set_xlabel('Expression Level (Z-score)', fontsize=16)

        # Title for bar chart section
        ax_bars.set_title(f'Top Molecular Subclasses', fontsize=18, pad=20)

        ax_bars.grid(axis='x', linestyle='--', alpha=0.3)

        # Add legend well above the plot
        ax_bars.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=16)

        # Extend x-axis to make room for annotations
        ax_bars.set_xlim(min(min(r1_values), min(r2_values)) * 1.2, annotation_pos * 1.5)

        # Add explanation text with more breathing room
        explanation = (
            f"Bar chart shows the top {len(labels)} molecular subclasses. "
            f"Full-vector comparison shows {full_similarity:.2f} similarity."
        )

        ax_bars.text(0.5, -0.15, explanation,
                     transform=ax_bars.transAxes,
                     fontsize=14, ha='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Now copy contents to our target axis
        # This is tricky since we've been working with a temporary figure
        # We'll save it to a buffer and then draw it in our target axis
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)

        # Now draw this image in our target axis
        from matplotlib.image import imread
        img = imread(buf)
        ax.imshow(img)
        ax.axis('off')

        plt.close(fig)  # Close the temporary figure

        return metrics

    def export_molecular_metrics(self, output_dir):
        """
        Export recorded molecular analysis metrics to CSV files and visualization

        Parameters:
            output_dir: Directory to save the output files
        """
        if not hasattr(self, 'molecular_analysis_metrics') or not self.molecular_analysis_metrics:
            print("No molecular analysis metrics available to export.")
            return

        output_dir = Path(output_dir)

        # Convert to DataFrame for easy export
        metrics_df = pd.DataFrame([
            {k: v for k, v in m.items() if not isinstance(v, dict)}
            for m in self.molecular_analysis_metrics
        ])

        # Save to CSV
        output_path = output_dir / "molecular_analysis_metrics.csv"
        metrics_df.to_csv(output_path, index=False)
        print(f"Exported molecular metrics to {output_path}")

        # Create a summary file with just the key metrics
        summary_df = pd.DataFrame({
            'region1': [m['region1_acronym'] for m in self.molecular_analysis_metrics],
            'region2': [m['region2_acronym'] for m in self.molecular_analysis_metrics],
            'full_vector_similarity': [m['full_vector_similarity'] for m in self.molecular_analysis_metrics],
            'selected_subclass_similarity': [m['selected_subclass_similarity'] for m in
                                             self.molecular_analysis_metrics],
            'pca_distance': [m['pca_distance'] for m in self.molecular_analysis_metrics],
            'pca_variance_explained_1': [m['pca_variance_explained'][0] for m in self.molecular_analysis_metrics],
            'pca_variance_explained_2': [m['pca_variance_explained'][1] for m in self.molecular_analysis_metrics]
        })

        summary_path = output_dir / "molecular_similarity_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Exported similarity summary to {summary_path}")

        # Generate a standalone PCA visualization of all compared regions
        if len(self.molecular_analysis_metrics) > 1:
            self._export_pca_overview(output_dir)

    def _export_pca_overview(self, output_dir):
        """Create a standalone visualization of all PCA results"""
        # Extract all PCA coordinates
        all_regions = {}
        for m in self.molecular_analysis_metrics:
            for region, coords in m['pca_coordinates'].items():
                if region not in all_regions:
                    all_regions[region] = coords

        # Create a special figure showing all regions in PCA space
        fig, ax = plt.subplots(figsize=(12, 10))

        # Get a color cycle for different regions
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_regions)))

        # Plot each region
        for i, (region, coords) in enumerate(all_regions.items()):
            ax.scatter(coords[0], coords[1], c=[colors[i]], s=120, label=region, edgecolor='k')

        # Add region labels
        for region, coords in all_regions.items():
            ax.annotate(region, (coords[0], coords[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold')

        # Calculate average variance explained across all analyses
        avg_var_explained = [
            np.mean([m['pca_variance_explained'][0] for m in self.molecular_analysis_metrics]),
            np.mean([m['pca_variance_explained'][1] for m in self.molecular_analysis_metrics])
        ]

        # Add labels and styling
        ax.set_xlabel(f'PC1 ({avg_var_explained[0]:.1%} variance explained)', fontsize=14)
        ax.set_ylabel(f'PC2 ({avg_var_explained[1]:.1%} variance explained)', fontsize=14)
        ax.set_title('Molecular Composition PCA Overview', fontsize=16, fontweight='bold')
        ax.grid(alpha=0.3)

        # Add legend if not too many regions
        if len(all_regions) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the figure
        output_path = output_dir / "molecular_pca_overview.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Exported PCA overview to {output_path}")

    def _create_distance_heatmaps_improved(self, ax_morph, ax_mol, r1_id, r2_id, r1_acro, r2_acro):
        """Create comparative distance heatmaps with better spacing and separate axes"""
        # Get nearby regions for comparison
        r1_distances = self.morph_dist_matrix[r1_id].sort_values()
        r2_distances = self.morph_dist_matrix[r2_id].sort_values()

        # Get top 6 closest regions to each (reduced from 8 for better spacing)
        r1_closest = r1_distances.index[:6].tolist()
        r2_closest = r2_distances.index[:6].tolist()

        # Combine and remove duplicates
        comparison_regions = list(set(r1_closest + r2_closest))
        if len(comparison_regions) > 10:  # Reduced from 12 for better spacing
            comparison_regions = comparison_regions[:10]

        # Add the two regions themselves
        if r1_id not in comparison_regions:
            comparison_regions.append(r1_id)
        if r2_id not in comparison_regions:
            comparison_regions.append(r2_id)

        # Get region acronyms
        region_acronyms = [self.region_info.loc[r, 'acronym'] for r in comparison_regions]

        # Create distance matrices for visualization
        morph_submatrix = self.norm_morph_dist.loc[comparison_regions, comparison_regions]
        subclass_submatrix = self.norm_subclass_dist.loc[comparison_regions, comparison_regions]

        # Create custom colormaps with clear separation
        morph_cmap = LinearSegmentedColormap.from_list('morph', ['white', COLOR_PALETTE['region1']])
        subclass_cmap = LinearSegmentedColormap.from_list('subclass', ['white', COLOR_PALETTE['region2']])

        # Plot morphological distances with its own colorbar
        sns.heatmap(morph_submatrix, ax=ax_morph, cmap=morph_cmap,
                    xticklabels=region_acronyms, yticklabels=region_acronyms,
                    cbar_kws={'label': 'Morphological Distance'})

        # Plot molecular distances with its own colorbar
        sns.heatmap(subclass_submatrix, ax=ax_mol, cmap=subclass_cmap,
                    xticklabels=region_acronyms, yticklabels=region_acronyms,
                    cbar_kws={'label': 'Molecular Distance'})

        # Find positions of our case regions
        r1_idx = comparison_regions.index(r1_id)
        r2_idx = comparison_regions.index(r2_id)

        # Highlight the case regions in both heatmaps
        for ax_i in [ax_morph, ax_mol]:
            # Add a rectangle around the case pair cell
            rect = plt.Rectangle((r2_idx, r1_idx), 1, 1, fill=False, edgecolor='black', lw=2)
            ax_i.add_patch(rect)
            rect = plt.Rectangle((r1_idx, r2_idx), 1, 1, fill=False, edgecolor='black', lw=2)
            ax_i.add_patch(rect)

            # Adjust tick labels for clarity
            ax_i.set_xticklabels(region_acronyms, rotation=45, ha='right', fontsize=10)
            ax_i.set_yticklabels(region_acronyms, fontsize=10)

        # Add titles
        ax_morph.set_title('Morphological Distances', fontsize=16, pad=15)
        ax_mol.set_title('Molecular Distances', fontsize=16, pad=15)

        # Add explanation text far below the heatmaps
        ax_morph.text(0.5, -0.2,
                      "Darker colors indicate larger distances.\nThe black boxes highlight the specific region pair being analyzed.",
                      transform=ax_morph.transAxes, fontsize=12, ha='center',
                      bbox=dict(facecolor='white', alpha=0.8))

    def _create_3d_feature_visualization_improved(self, ax, r1_id, r2_id, r1_acro, r2_acro):
        """Create 3D visualization with improved spacing"""
        # Run PCA on all vectors to find key components
        all_morph = self.morph_vectors.values

        # Run PCA to get principal components
        morph_pca = PCA(n_components=2).fit(all_morph)

        # Get coordinates in PC space
        r1_morph = self.morph_vectors.loc[r1_id].values
        r2_morph = self.morph_vectors.loc[r2_id].values

        r1_pc = morph_pca.transform([r1_morph])[0]
        r2_pc = morph_pca.transform([r2_morph])[0]

        # Also get a molecular signature via PCA
        subclass_sample = self.subclass_vectors.values[:, :min(100, self.subclass_vectors.shape[1])]
        mol_pca = PCA(n_components=1).fit(subclass_sample)

        r1_subclass = self.subclass_vectors.loc[r1_id].values[:min(100, self.subclass_vectors.shape[1])]
        r2_subclass = self.subclass_vectors.loc[r2_id].values[:min(100, self.subclass_vectors.shape[1])]

        r1_mol_pc = mol_pca.transform([r1_subclass])[0]
        r2_mol_pc = mol_pca.transform([r2_subclass])[0]

        # Get fewer pairs for context to reduce crowding
        top_mismatch_pairs = [
            (self.significant_pairs[i]['region1_id'], self.significant_pairs[i]['region2_id'])
            for i in range(min(15, len(self.significant_pairs)))  # Reduced from 30
        ]

        # Collect all points for the 3D scatter plot
        x_all, y_all, z_all = [], [], []
        colors, sizes = [], []

        # Add points for all significant mismatch pairs
        for r1, r2 in top_mismatch_pairs:
            # Get morphological principal components
            r1_morph_data = self.morph_vectors.loc[r1].values
            r2_morph_data = self.morph_vectors.loc[r2].values

            r1_morph_pc = morph_pca.transform([r1_morph_data])[0]
            r2_morph_pc = morph_pca.transform([r2_morph_data])[0]

            # Get molecular principal component
            r1_subclass_data = self.subclass_vectors.loc[r1].values[:min(100, self.subclass_vectors.shape[1])]
            r2_subclass_data = self.subclass_vectors.loc[r2].values[:min(100, self.subclass_vectors.shape[1])]

            r1_mol_pc_val = mol_pca.transform([r1_subclass_data])[0]
            r2_mol_pc_val = mol_pca.transform([r2_subclass_data])[0]

            # Add to lists
            x_all.extend([r1_morph_pc[0], r2_morph_pc[0]])
            y_all.extend([r1_morph_pc[1], r2_morph_pc[1]])
            z_all.extend([r1_mol_pc_val[0], r2_mol_pc_val[0]])

            # Gray dots for context, smaller size
            colors.extend(['gray', 'gray'])
            sizes.extend([30, 30])

        # Add the current case pair with special highlighting - larger size
        x_all.extend([r1_pc[0], r2_pc[0]])
        y_all.extend([r1_pc[1], r2_pc[1]])
        z_all.extend([r1_mol_pc[0], r2_mol_pc[0]])
        colors.extend([COLOR_PALETTE['region1'], COLOR_PALETTE['region2']])
        sizes.extend([300, 300])  # Increased from 200

        # Create 3D scatter plot with context
        scatter = ax.scatter(x_all, y_all, z_all, c=colors, s=sizes, alpha=0.7, edgecolors='w', linewidth=0.5)

        # Add a line connecting the case pair
        ax.plot([r1_pc[0], r2_pc[0]], [r1_pc[1], r2_pc[1]], [r1_mol_pc[0], r2_mol_pc[0]],
                'k-', alpha=0.7, linewidth=2)

        # Add labels for the case pair - positioned with large offset to avoid overlap
        ax.text(r1_pc[0], r1_pc[1], r1_mol_pc[0] + 1.0, r1_acro, fontsize=14, fontweight='bold',
                color=COLOR_PALETTE['region1'])
        ax.text(r2_pc[0], r2_pc[1], r2_mol_pc[0] + 1.0, r2_acro, fontsize=14, fontweight='bold',
                color=COLOR_PALETTE['region2'])

        # Set labels
        ax.set_xlabel('Morphology PC1', fontsize=14, labelpad=15)
        ax.set_ylabel('Morphology PC2', fontsize=14, labelpad=15)
        ax.set_zlabel('Molecular PC1', fontsize=14, labelpad=15)

        # Title - positioned high above the plot
        ax.set_title('3D Feature Space Visualization', fontsize=18, fontweight='bold', pad=30)

        # Improve 3D plot appearance
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Add explanation text as separate annotation below the figure
        explanation = (
            "This 3D visualization shows the position of these regions in a combined\n"
            "morphological (x,y) and molecular (z) feature space.\n"
            "Gray points show other region pairs for context."
        )

        # Using text positioned far below the visualization
        ax.text2D(0.5, -0.2, explanation, fontsize=12, ha='center', va='top',
                  transform=ax.transAxes,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Adjust view angle for better visualization
        ax.view_init(elev=30, azim=45)

        # Expand axis limits by 20% for more space
        x_range = max(x_all) - min(x_all)
        y_range = max(y_all) - min(y_all)
        z_range = max(z_all) - min(z_all)

        ax.set_xlim(min(x_all) - 0.2 * x_range, max(x_all) + 0.2 * x_range)
        ax.set_ylim(min(y_all) - 0.2 * y_range, max(y_all) + 0.2 * y_range)
        ax.set_zlim(min(z_all) - 0.2 * z_range, max(z_all) + 0.2 * z_range)

    def _create_metrics_comparison_improved(self, ax, case):
        """Create comparison of similarity metrics with circular visualization and improved spacing"""
        # Extract metrics
        morph_similarity = 1 - case['morph_distance']
        molecular_similarity = 1 - case['subclass_distance']
        mismatch = case['mismatch']

        # Create a circular visualization of similarities
        # Using two semi-circles for the two similarity types

        # Create a circle with two parts
        morph_theta = np.linspace(0, np.pi, 100)
        mol_theta = np.linspace(np.pi, 2 * np.pi, 100)

        r = 0.7  # radius of the circle - slightly reduced to allow more space

        # Calculate coordinates for the two semi-circles
        morph_x = r * np.cos(morph_theta)
        morph_y = r * np.sin(morph_theta)
        mol_x = r * np.cos(mol_theta)
        mol_y = r * np.sin(mol_theta)

        # Plot the circle outline
        ax.plot(np.concatenate([morph_x, mol_x]), np.concatenate([morph_y, mol_y]),
                'k-', alpha=0.3, linewidth=1)

        # Fill the semicircles according to the similarity values
        # Map similarity from [0,1] to [0,r]
        morph_r = r * morph_similarity
        mol_r = r * molecular_similarity

        # Create wedges
        morph_wedge = Wedge((0, 0), morph_r, 0, 180, width=0.05,
                            fc=COLOR_PALETTE['region1'], alpha=0.7)
        mol_wedge = Wedge((0, 0), mol_r, 180, 360, width=0.05,
                          fc=COLOR_PALETTE['region2'], alpha=0.7)

        ax.add_patch(morph_wedge)
        ax.add_patch(mol_wedge)

        # Add similarity values as text - positioned far from the center
        ax.text(0, morph_r / 1.3, f"{morph_similarity:.2f}",
                ha='center', va='center', fontsize=16, fontweight='bold',
                color=COLOR_PALETTE['region1'])

        ax.text(0, -mol_r / 1.3, f"{molecular_similarity:.2f}",
                ha='center', va='center', fontsize=16, fontweight='bold',
                color=COLOR_PALETTE['region2'])

        # Add a central mismatch indicator
        mismatch_circle = Circle((0, 0), mismatch * r * 0.8,
                                 fc=COLOR_PALETTE['mismatch'], alpha=0.7)
        ax.add_patch(mismatch_circle)

        # Add mismatch value in center
        ax.text(0, 0, f"{mismatch:.2f}",
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if mismatch > 0.5 else 'black')

        # Add labels - positioned very far outside the circle
        ax.text(0, r * 1.5, "Morphological Similarity",
                ha='center', va='bottom', fontsize=16, fontweight='bold',
                color=COLOR_PALETTE['region1'])

        ax.text(0, -r * 1.5, "Molecular Similarity",
                ha='center', va='top', fontsize=16, fontweight='bold',
                color=COLOR_PALETTE['region2'])

        # Add mismatch label to the side with plenty of space
        ax.text(r * 1.6, 0, "Mismatch\nIndex",
                ha='left', va='center', fontsize=16, fontweight='bold',
                color=COLOR_PALETTE['mismatch'])

        # Set equal aspect ratio and generous limits
        ax.set_aspect('equal')
        ax.set_xlim(-r * 2.0, r * 2.0)  # Expanded from 1.8
        ax.set_ylim(-r * 2.0, r * 2.0)  # Expanded from 1.8

        # Remove ticks and frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add title with generous padding
        ax.set_title('Similarity & Mismatch Metrics', fontsize=18, fontweight='bold', pad=30)

def load_data(vector_dir):
    """
    Load vector data from CSV files

    Parameters:
        vector_dir: Directory containing vector data files

    Returns:
        tuple: (morphological vectors, subclass vectors, region info)
    """
    try:
        # Load standardized vectors
        morph_vectors = pd.read_csv(f"{vector_dir}/standardized_morphological_vectors.csv", index_col=0)
        subclass_vectors = pd.read_csv(f"{vector_dir}/standardized_subclass_vectors.csv", index_col=0)

        # Load region info
        region_info = pd.read_csv(f"{vector_dir}/region_info.csv", index_col=0)

        # If region_info doesn't have a name column, create one from acronyms
        if 'name' not in region_info.columns:
            region_info['name'] = region_info['acronym']

        print(f"Loaded data: {len(morph_vectors)} regions with {morph_vectors.shape[1]} morphological features")
        print(f"               and {subclass_vectors.shape[1]} subclass features")

        return morph_vectors, subclass_vectors, region_info

    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Checking files in {vector_dir}...")

        vector_path = Path(vector_dir)
        if vector_path.exists():
            print("Files in directory:")
            for file in vector_path.iterdir():
                print(f"  {file.name}")
        else:
            print(f"Directory {vector_dir} does not exist")

        raise

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Brain Region Morphology-Molecular Mismatch Analyzer')
    parser.add_argument('--data_dir', type=str, default='./knowledge_graph_v5/vector_analysis',
                        help='Directory containing vector data files')
    parser.add_argument('--output_dir', type=str, default='./mismatch_analysisV2',
                        help='Directory for saving output files')
    parser.add_argument('--min_data_percentile', type=int, default=30,
                        help='Minimum data completeness percentile threshold (0-100)')

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Analysis started. Loading data from {args.data_dir}")
    print(f"Results will be saved to {output_dir}")
    print(f"Using minimum data completeness percentile: {args.min_data_percentile}")

    # Load data
    morph_vectors, subclass_vectors, region_info = load_data(args.data_dir)

    # Initialize analyzer with data quality threshold
    analyzer = MismatchAnalyzer(morph_vectors, subclass_vectors, region_info,
                                min_data_percentile=args.min_data_percentile)

    # 1. Create global relationship plot
    print("Creating global relationship plot...")
    global_fig = analyzer.plot_global_relationship()
    global_fig_path = output_dir / "global_structure_function_relationship.png"
    global_fig.savefig(global_fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved global relationship plot to {global_fig_path}")

    # 2. Create mismatch heatmap
    print("Creating mismatch heatmap...")
    heatmap_fig = analyzer.plot_mismatch_heatmap()
    if heatmap_fig is not None:
        heatmap_fig_path = output_dir / "region_mismatch_heatmap.png"
        heatmap_fig.savefig(heatmap_fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved mismatch heatmap to {heatmap_fig_path}")

    # 3. Create region mismatch profile plot
    print("Creating region mismatch profile...")
    profile_fig = analyzer.plot_region_mismatch_profile(top_n=15)
    profile_fig_path = output_dir / "top_mismatch_regions.png"
    profile_fig.savefig(profile_fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved region mismatch profile to {profile_fig_path}")

    # 4. Create case studies
    print("Creating case studies...")
    case_figs = analyzer.create_case_study_visualization(case_type='both', n_cases=3)
    if case_figs:
        # Save each case study as a separate file
        for i, fig in enumerate(case_figs):
            case_fig_path = output_dir / f"case_study_{i + 1}.png"
            fig.savefig(case_fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved case study {i + 1} to {case_fig_path}")

    # 5. Save mismatch data to CSV
    print("Saving mismatch data to CSV...")

    # Save region mismatch indices
    region_mismatch_df = pd.DataFrame({
        'region_id': list(analyzer.region_mismatch_index.keys()),
        'acronym': [region_info.loc[r, 'acronym'] for r in analyzer.region_mismatch_index.keys()],
        'data_quality': [region_info.loc[r, 'data_quality'] for r in analyzer.region_mismatch_index.keys()],
        'mismatch_index': list(analyzer.region_mismatch_index.values())
    }).sort_values('mismatch_index', ascending=False)

    region_mismatch_path = output_dir / "region_mismatch_indices.csv"
    region_mismatch_df.to_csv(region_mismatch_path, index=False)

    # Save significant pairs
    pairs_df = pd.DataFrame(analyzer.significant_pairs)
    pairs_path = output_dir / "significant_mismatch_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)

    print("Analysis complete!")
    print(f"All results saved to {output_dir}")
    print("Summary of key findings:")
    print(f"- Analyzed {len(analyzer.data_rich_regions)} regions with sufficient data")
    print(f"- Found {len(analyzer.significant_pairs)} significant mismatch pairs")

    # Extract top regions
    top_regions = region_mismatch_df.head(5)
    print("\nTop 5 regions with highest mismatch index:")
    for _, row in top_regions.iterrows():
        print(f"  {row['acronym']}: {row['mismatch_index']:.3f}")

    # Extract top pairs
    if len(analyzer.significant_pairs) > 0:
        print("\nTop 3 mismatched region pairs:")
        for i in range(min(3, len(analyzer.significant_pairs))):
            pair = analyzer.significant_pairs[i]
            print(f"  {pair['region1_acronym']}-{pair['region2_acronym']}: {pair['mismatch']:.3f} "
                  f"({'Morph similar' if pair['type'] == 'morph_similar' else 'Mol similar'})")

if __name__ == "__main__":
    main()