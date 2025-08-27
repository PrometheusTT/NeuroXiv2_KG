"""
NeuroXiv 2.0 Region Visualization Analyzer
- Calculates morphological and molecular composition differences
- Generates comparative visualizations

Author: wangmajortom
Date: 2025-08-27
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from collections import Counter, defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
sns.set_context("paper", font_scale=1.5)

# Constants
MORPH_ATTRIBUTES = [
    'axonal_bifurcation_remote_angle', 'axonal_branches', 'axonal_length',
    'axonal_maximum_branch_order', 'dendritic_bifurcation_remote_angle',
    'dendritic_branches', 'dendritic_length', 'dendritic_maximum_branch_order'
]

# Attribute labels for radar chart
MORPH_ATTRIBUTE_LABELS = {
    'axonal_bifurcation_remote_angle': 'Axonal Bifurcation Angle',
    'axonal_branches': 'Axonal Branches',
    'axonal_length': 'Axonal Length',
    'axonal_maximum_branch_order': 'Axonal Max Branch Order',
    'dendritic_bifurcation_remote_angle': 'Dendritic Bifurcation Angle',
    'dendritic_branches': 'Dendritic Branches',
    'dendritic_length': 'Dendritic Length',
    'dendritic_maximum_branch_order': 'Dendritic Max Branch Order'
}

# Custom color map
region_colors = {
    'V1': '#E41A1C',  # Red
    'MOs': '#377EB8',  # Blue
    'SSp': '#4DAF4A',  # Green
    'TH': '#984EA3',  # Purple
    'MOp': '#FF7F00',  # Orange
    'SSs': '#FFFF33',  # Yellow
    'ACA': '#A65628',  # Brown
    'AI': '#F781BF',  # Pink
    'HIP': '#999999',  # Gray
    'STR': '#66C2A5',  # Cyan
}


class RegionAnalyzer:
    """Region Analyzer - Calculate distances between brain regions in morphology and molecular composition"""

    def __init__(self, data_dir: Path):
        """Initialize analyzer, load data"""
        self.data_dir = Path(data_dir)
        self.regions_df = None
        self.has_subclass_df = None
        self.subclass_df = None
        self.morph_dist_matrix = None
        self.subclass_dist_matrix = None
        self.region_id_to_acronym = {}
        self.region_acronym_to_id = {}
        self.region_pairs = []

        # Ensure output directory exists
        self.output_dir = self.data_dir / "analysis_results"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self._load_data()

    def _load_data(self):
        """Load CSV data"""
        print("Loading data...")

        # Load region data
        regions_file = self.data_dir / "nodes" / "regions.csv"
        if regions_file.exists():
            self.regions_df = pd.read_csv(regions_file)
            print(f"Loaded {len(self.regions_df)} regions")

            # Create ID-acronym mapping
            if 'region_id:ID(Region)' in self.regions_df.columns and 'acronym' in self.regions_df.columns:
                self.region_id_to_acronym = dict(zip(
                    self.regions_df['region_id:ID(Region)'],
                    self.regions_df['acronym']
                ))
                self.region_acronym_to_id = dict(zip(
                    self.regions_df['acronym'],
                    self.regions_df['region_id:ID(Region)']
                ))
                print(f"Created {len(self.region_id_to_acronym)} region ID-acronym mappings")
        else:
            print(f"Warning: Region file does not exist: {regions_file}")

        # Load subclass data
        subclass_file = self.data_dir / "nodes" / "subclass.csv"
        if subclass_file.exists():
            self.subclass_df = pd.read_csv(subclass_file)
            print(f"Loaded {len(self.subclass_df)} subclasses")
        else:
            print(f"Warning: Subclass file does not exist: {subclass_file}")

        # Load region-subclass relationships
        has_subclass_file = self.data_dir / "relationships" / "has_subclass.csv"
        if has_subclass_file.exists():
            self.has_subclass_df = pd.read_csv(has_subclass_file)
            print(f"Loaded {len(self.has_subclass_df)} region-subclass relationships")
        else:
            print(f"Warning: has_subclass file does not exist: {has_subclass_file}")

    def calculate_distance_matrices(self):
        """Calculate distance matrices for morphological features and molecular composition"""
        print("Calculating distance matrices...")

        # 1. Calculate morphological feature distance matrix
        morph_columns = [f"{attr}:float" for attr in MORPH_ATTRIBUTES]

        # Check if all morphological feature columns exist
        missing_cols = [col for col in morph_columns if col not in self.regions_df.columns]
        if missing_cols:
            print(f"Warning: Missing morphology feature columns: {missing_cols}")
            morph_columns = [col for col in morph_columns if col in self.regions_df.columns]

        if morph_columns:
            # Extract and standardize morphological features
            morph_features = self.regions_df[morph_columns].copy()
            morph_features.columns = [col.replace(':float', '') for col in morph_columns]

            # Replace NaN with 0
            morph_features.fillna(0, inplace=True)

            # Standardize
            scaler = StandardScaler()
            morph_features_scaled = scaler.fit_transform(morph_features)

            # Calculate distance matrix
            morph_dist = pdist(morph_features_scaled, metric='euclidean')
            self.morph_dist_matrix = squareform(morph_dist)

            print(f"Calculated morphology feature distance matrix (shape: {self.morph_dist_matrix.shape})")
        else:
            print("Error: Cannot calculate morphology distance matrix - missing necessary columns")
            return False

        # 2. Calculate molecular composition distance matrix
        if self.has_subclass_df is not None:
            # Create region-subclass matrix
            subclass_matrix = pd.pivot_table(
                self.has_subclass_df,
                values='pct_cells:float',
                index=':START_ID(Region)',
                columns=':END_ID(Subclass)',
                fill_value=0
            )

            # Calculate distance matrix
            subclass_dist = pdist(subclass_matrix.values, metric='cosine')
            self.subclass_dist_matrix = squareform(subclass_dist)

            # Save region ID order for later reference
            self.subclass_region_ids = subclass_matrix.index.tolist()

            print(f"Calculated molecular composition distance matrix (shape: {self.subclass_dist_matrix.shape})")

            # Record mapping from region ID to index for later queries
            self.region_id_to_index = {region_id: i for i, region_id in enumerate(self.subclass_region_ids)}

            # Create region pair list (only including regions with subclass data)
            region_ids = subclass_matrix.index.tolist()
            self.region_pairs = [(i, j) for i in range(len(region_ids))
                                 for j in range(i + 1, len(region_ids))]

            print(f"Created {len(self.region_pairs)} region pairs")
            return True
        else:
            print("Error: Cannot calculate molecular composition distance matrix - missing has_subclass data")
            return False

    def plot_distance_relationship(self):
        """Plot A: Scatter plot of morphological feature distance vs molecular composition distance"""
        if self.morph_dist_matrix is None or self.subclass_dist_matrix is None:
            print("Error: Distance matrices not calculated")
            return

        print("Plotting distance relationship scatter plot...")

        # Extract distances for each region pair
        morph_distances = []
        subclass_distances = []
        pair_labels = []

        for i, j in self.region_pairs:
            region_id1 = self.subclass_region_ids[i]
            region_id2 = self.subclass_region_ids[j]

            # Get region acronyms
            acronym1 = self.region_id_to_acronym.get(region_id1, f"Region_{region_id1}")
            acronym2 = self.region_id_to_acronym.get(region_id2, f"Region_{region_id2}")

            morph_distances.append(self.morph_dist_matrix[i, j])
            subclass_distances.append(self.subclass_dist_matrix[i, j])
            pair_labels.append(f"{acronym1}-{acronym2}")

        # Create scatter plot
        plt.figure(figsize=(12, 10))

        # Calculate correlation coefficient
        correlation = np.corrcoef(morph_distances, subclass_distances)[0, 1]

        # Create DataFrame for seaborn
        scatter_df = pd.DataFrame({
            'Morphological Distance': morph_distances,
            'Molecular Composition Distance': subclass_distances,
            'Region Pair': pair_labels
        })

        # Plot scatter plot using seaborn
        sns.scatterplot(
            data=scatter_df,
            x='Morphological Distance',
            y='Molecular Composition Distance',
            alpha=0.7,
            s=50
        )

        # Fit regression line
        sns.regplot(
            data=scatter_df,
            x='Morphological Distance',
            y='Molecular Composition Distance',
            scatter=False,
            color='red'
        )

        plt.title(f'Relationship between Morphological and Molecular Distances\n(Correlation: {correlation:.3f})',
                  fontsize=16)
        plt.xlabel('Morphological Distance (Euclidean)', fontsize=14)
        plt.ylabel('Molecular Composition Distance (Cosine)', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Save image
        output_file = self.output_dir / "distance_relationship.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to: {output_file}")
        plt.close()

        # Identify special region pairs
        return self._identify_special_pairs(scatter_df)

    def _identify_special_pairs(self, scatter_df):
        """Identify special region pairs:
        1. Similar morphology but different molecular composition
        2. Different morphology but similar molecular composition
        """
        print("Identifying special region pairs...")

        # Calculate medians
        morph_median = np.median(scatter_df['Morphological Distance'])
        subclass_median = np.median(scatter_df['Molecular Composition Distance'])

        # Categorize region pairs based on distance medians
        scatter_df['Type'] = 'average'

        # Similar morphology but different molecular composition
        similar_morph_diff_subclass = scatter_df[
            (scatter_df['Morphological Distance'] < morph_median) &
            (scatter_df['Molecular Composition Distance'] > subclass_median * 1.2)
            ].sort_values('Molecular Composition Distance', ascending=False).head(5)
        similar_morph_diff_subclass['Type'] = 'similar_morph_diff_subclass'

        # Different morphology but similar molecular composition
        diff_morph_similar_subclass = scatter_df[
            (scatter_df['Morphological Distance'] > morph_median * 1.2) &
            (scatter_df['Molecular Composition Distance'] < subclass_median)
            ].sort_values('Morphological Distance', ascending=False).head(5)
        diff_morph_similar_subclass['Type'] = 'diff_morph_similar_subclass'

        # Combine special pairs
        special_pairs = pd.concat([similar_morph_diff_subclass, diff_morph_similar_subclass])

        # Output special region pairs
        if not special_pairs.empty:
            print("\nSpecial region pairs:")
            for i, row in special_pairs.iterrows():
                print(f"  {row['Region Pair']}: Morph Distance={row['Morphological Distance']:.3f}, "
                      f"Molecular Distance={row['Molecular Composition Distance']:.3f}, Type={row['Type']}")

            # Save special region pairs
            special_pairs_file = self.output_dir / "special_region_pairs.csv"
            special_pairs.to_csv(special_pairs_file, index=False)
            print(f"Saved special region pairs to: {special_pairs_file}")

        return special_pairs

    def plot_morphology_radar(self, region_pairs):
        """Plot B: Morphological feature radar charts for selected region pairs"""
        if region_pairs.empty:
            print("Warning: No region pairs to plot radar charts")
            return

        print("Plotting morphological feature radar charts...")

        # Plot radar chart for each region pair
        for i, row in region_pairs.iterrows():
            region_pair = row['Region Pair']
            pair_type = row['Type']

            # Parse region pair
            region1_acronym, region2_acronym = region_pair.split('-')

            # Get region IDs
            region1_id = self.region_acronym_to_id.get(region1_acronym)
            region2_id = self.region_acronym_to_id.get(region2_acronym)

            if region1_id is None or region2_id is None:
                print(f"Warning: Cannot find region IDs: {region_pair}")
                continue

            # Get morphological features
            region1_data = self.regions_df[self.regions_df['region_id:ID(Region)'] == region1_id]
            region2_data = self.regions_df[self.regions_df['region_id:ID(Region)'] == region2_id]

            if region1_data.empty or region2_data.empty:
                print(f"Warning: Cannot find region data: {region_pair}")
                continue

            # Extract morphological features
            morph_columns = [f"{attr}:float" for attr in MORPH_ATTRIBUTES]

            # Check if all morphological feature columns exist
            missing_cols = [col for col in morph_columns if col not in self.regions_df.columns]
            if missing_cols:
                print(f"Warning: Missing morphology feature columns: {missing_cols}")
                morph_columns = [col for col in morph_columns if col in self.regions_df.columns]

            # Get feature values and standardize
            region1_features = region1_data[morph_columns].iloc[0].values
            region2_features = region2_data[morph_columns].iloc[0].values

            # Standardize
            combined_features = np.vstack([region1_features, region2_features])
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(combined_features)

            region1_norm = normalized_features[0]
            region2_norm = normalized_features[1]

            # Set up radar chart parameters
            categories = [MORPH_ATTRIBUTE_LABELS.get(attr, attr) for attr in MORPH_ATTRIBUTES]
            N = len(categories)

            # Set angles
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the radar chart

            # Extend feature vectors to close the radar chart
            region1_norm = np.append(region1_norm, region1_norm[0])
            region2_norm = np.append(region2_norm, region2_norm[0])

            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

            # Plot first region
            ax.plot(angles, region1_norm, 'o-', linewidth=2, label=region1_acronym,
                    color=region_colors.get(region1_acronym, '#1f77b4'))
            ax.fill(angles, region1_norm, alpha=0.1, color=region_colors.get(region1_acronym, '#1f77b4'))

            # Plot second region
            ax.plot(angles, region2_norm, 'o-', linewidth=2, label=region2_acronym,
                    color=region_colors.get(region2_acronym, '#ff7f0e'))
            ax.fill(angles, region2_norm, alpha=0.1, color=region_colors.get(region2_acronym, '#ff7f0e'))

            # Set tick labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)

            # Add y-axis grid lines and labels
            ax.set_rlabel_position(0)
            ax.tick_params(axis='y', labelsize=10)

            # Add title
            pair_type_label = "Similar Morphology, Different Composition" if pair_type == "similar_morph_diff_subclass" else "Different Morphology, Similar Composition"
            plt.title(f"{region1_acronym} vs {region2_acronym}\n{pair_type_label}", fontsize=16, pad=20)

            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            # Save image
            output_file = self.output_dir / f"morphology_radar_{region1_acronym}_vs_{region2_acronym}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved radar chart to: {output_file}")
            plt.close()

    def plot_subclass_composition(self, region_pairs):
        """Plot C: Subclass composition bar charts for selected region pairs"""
        if region_pairs.empty or self.has_subclass_df is None:
            print("Warning: No region pairs or subclass data to plot bar charts")
            return

        print("Plotting subclass composition bar charts...")

        # Plot bar chart for each region pair
        for i, row in region_pairs.iterrows():
            region_pair = row['Region Pair']
            pair_type = row['Type']

            # Parse region pair
            region1_acronym, region2_acronym = region_pair.split('-')

            # Get region IDs
            region1_id = self.region_acronym_to_id.get(region1_acronym)
            region2_id = self.region_acronym_to_id.get(region2_acronym)

            if region1_id is None or region2_id is None:
                print(f"Warning: Cannot find region IDs: {region_pair}")
                continue

            # Get region subclass composition
            region1_subclasses = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region1_id]
            region2_subclasses = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region2_id]

            if region1_subclasses.empty or region2_subclasses.empty:
                print(f"Warning: Cannot find region subclass data: {region_pair}")
                continue

            # Get top 10 subclasses
            region1_top = region1_subclasses.sort_values('rank:int').head(10)
            region2_top = region2_subclasses.sort_values('rank:int').head(10)

            # Combine top 10 subclasses from both regions
            combined_subclasses = pd.concat([region1_top, region2_top])
            unique_subclass_ids = combined_subclasses[':END_ID(Subclass)'].unique()

            # Get subclass names
            subclass_names = {}
            if self.subclass_df is not None:
                for subclass_id in unique_subclass_ids:
                    subclass_row = self.subclass_df[self.subclass_df['tran_id:ID(Subclass)'] == subclass_id]
                    if not subclass_row.empty:
                        subclass_names[subclass_id] = subclass_row.iloc[0]['name']
                    else:
                        subclass_names[subclass_id] = f"Subclass_{subclass_id}"
            else:
                subclass_names = {subclass_id: f"Subclass_{subclass_id}" for subclass_id in unique_subclass_ids}

            # Create data for plotting
            plot_data = []
            for subclass_id in unique_subclass_ids:
                # Get data for region 1
                region1_row = region1_subclasses[region1_subclasses[':END_ID(Subclass)'] == subclass_id]
                pct1 = region1_row['pct_cells:float'].iloc[0] if not region1_row.empty else 0

                # Get data for region 2
                region2_row = region2_subclasses[region2_subclasses[':END_ID(Subclass)'] == subclass_id]
                pct2 = region2_row['pct_cells:float'].iloc[0] if not region2_row.empty else 0

                plot_data.append({
                    'subclass_id': subclass_id,
                    'subclass_name': subclass_names[subclass_id],
                    region1_acronym: pct1 * 100,  # Convert to percentage
                    region2_acronym: pct2 * 100  # Convert to percentage
                })

            # Convert to DataFrame
            plot_df = pd.DataFrame(plot_data)

            # Sort by total percentage in both regions
            plot_df['total_pct'] = plot_df[region1_acronym] + plot_df[region2_acronym]
            plot_df = plot_df.sort_values('total_pct', ascending=False).head(15)

            # Plot bar chart
            plt.figure(figsize=(14, 10))

            x = np.arange(len(plot_df))
            width = 0.35

            # Plot bars for both regions
            ax = plt.gca()
            bars1 = ax.bar(x - width / 2, plot_df[region1_acronym], width,
                           label=region1_acronym, color=region_colors.get(region1_acronym, '#1f77b4'))
            bars2 = ax.bar(x + width / 2, plot_df[region2_acronym], width,
                           label=region2_acronym, color=region_colors.get(region2_acronym, '#ff7f0e'))

            # Add labels and title
            ax.set_ylabel('Cell Percentage (%)', fontsize=14)
            ax.set_title(f"{region1_acronym} vs {region2_acronym} Subclass Composition", fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df['subclass_name'], rotation=45, ha='right', fontsize=12)
            ax.legend(fontsize=12)

            # Add grid lines
            ax.grid(axis='y', alpha=0.3)

            # Adjust layout
            plt.tight_layout()

            # Save image
            output_file = self.output_dir / f"subclass_composition_{region1_acronym}_vs_{region2_acronym}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved composition bar chart to: {output_file}")
            plt.close()

    def plot_neurotransmitter_distribution(self, region_pairs):
        """Plot neurotransmitter system distribution for selected region pairs"""
        if region_pairs.empty or self.has_subclass_df is None or self.subclass_df is None:
            print("Warning: Missing data for neurotransmitter analysis")
            return

        print("Plotting neurotransmitter distribution charts...")

        # Define neurotransmitter categories
        neurotransmitter_categories = [
            "Glutamatergic", "GABAergic", "Dopaminergic", "Serotonergic",
            "Cholinergic", "Peptidergic", "Noradrenergic", "Other"
        ]

        # Preprocess subclass data, extract neurotransmitter type
        self.subclass_df['dominant_neurotransmitter_type'] = self.subclass_df['dominant_neurotransmitter_type'].fillna(
            'Unknown')

        # Simplify neurotransmitter type to main categories
        def simplify_neurotransmitter(nt_type):
            nt_type = str(nt_type).lower()
            if 'glutamatergic' in nt_type or 'glutamate' in nt_type:
                return "Glutamatergic"
            elif 'gaba' in nt_type:
                return "GABAergic"
            elif 'dopamine' in nt_type or 'dopaminergic' in nt_type:
                return "Dopaminergic"
            elif 'serotonin' in nt_type or '5-ht' in nt_type:
                return "Serotonergic"
            elif 'cholinergic' in nt_type or 'acetylcholine' in nt_type:
                return "Cholinergic"
            elif 'peptide' in nt_type or 'neuropeptide' in nt_type:
                return "Peptidergic"
            elif 'noradrenaline' in nt_type or 'norepinephrine' in nt_type:
                return "Noradrenergic"
            else:
                return "Other"

        # Apply neurotransmitter simplification
        self.subclass_df['neurotransmitter_category'] = self.subclass_df['dominant_neurotransmitter_type'].apply(
            simplify_neurotransmitter)

        # For each region pair, analyze neurotransmitter distribution
        for i, row in region_pairs.iterrows():
            region_pair = row['Region Pair']
            pair_type = row['Type']

            # Parse region pair
            region1_acronym, region2_acronym = region_pair.split('-')

            # Get region IDs
            region1_id = self.region_acronym_to_id.get(region1_acronym)
            region2_id = self.region_acronym_to_id.get(region2_acronym)

            if region1_id is None or region2_id is None:
                print(f"Warning: Cannot find region IDs: {region_pair}")
                continue

            # Get region subclass composition
            region1_subclasses = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region1_id]
            region2_subclasses = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region2_id]

            if region1_subclasses.empty or region2_subclasses.empty:
                print(f"Warning: Cannot find region subclass data: {region_pair}")
                continue

            # Calculate neurotransmitter distribution for each region
            region1_nt_dist = self._calculate_neurotransmitter_distribution(region1_subclasses,
                                                                            neurotransmitter_categories)
            region2_nt_dist = self._calculate_neurotransmitter_distribution(region2_subclasses,
                                                                            neurotransmitter_categories)

            # Create data for plotting
            data = []
            for nt_type in neurotransmitter_categories:
                data.append({
                    'Neurotransmitter Type': nt_type,
                    region1_acronym: region1_nt_dist.get(nt_type, 0) * 100,  # Convert to percentage
                    region2_acronym: region2_nt_dist.get(nt_type, 0) * 100,  # Convert to percentage
                    'Difference': region1_nt_dist.get(nt_type, 0) - region2_nt_dist.get(nt_type, 0)
                })

            df = pd.DataFrame(data)
            df = df.sort_values('Difference', ascending=False)

            # Plot neurotransmitter distribution comparison
            plt.figure(figsize=(12, 8))

            # Plot double bar chart
            ax = plt.gca()
            x = np.arange(len(df))
            width = 0.35

            bars1 = ax.bar(x - width / 2, df[region1_acronym], width,
                           label=region1_acronym, color=region_colors.get(region1_acronym, '#1f77b4'))
            bars2 = ax.bar(x + width / 2, df[region2_acronym], width,
                           label=region2_acronym, color=region_colors.get(region2_acronym, '#ff7f0e'))

            # Add labels and title
            ax.set_ylabel('Percentage (%)', fontsize=14)
            ax.set_title(f"{region1_acronym} vs {region2_acronym} Neurotransmitter System Distribution", fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(df['Neurotransmitter Type'], rotation=45, ha='right', fontsize=12)
            ax.legend(fontsize=12)

            # Add value labels
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

            add_labels(bars1)
            add_labels(bars2)

            # Add grid lines
            ax.grid(axis='y', alpha=0.3)

            # Adjust layout
            plt.tight_layout()

            # Save image
            output_file = self.output_dir / f"neurotransmitter_distribution_{region1_acronym}_vs_{region2_acronym}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved neurotransmitter distribution chart to: {output_file}")
            plt.close()

    def _calculate_neurotransmitter_distribution(self, region_subclasses, nt_categories):
        """Calculate neurotransmitter distribution in a region"""
        # Get percentage of each subclass in the region
        subclass_weights = {}
        for _, row in region_subclasses.iterrows():
            subclass_id = row[':END_ID(Subclass)']
            weight = row['pct_cells:float']
            subclass_weights[subclass_id] = weight

        # Get neurotransmitter type for each subclass
        subclass_nt = {}
        for _, row in self.subclass_df.iterrows():
            subclass_id = row['tran_id:ID(Subclass)']
            nt_type = row['neurotransmitter_category']
            subclass_nt[subclass_id] = nt_type

        # Calculate weighted percentage of each neurotransmitter type
        nt_distribution = {nt: 0 for nt in nt_categories}
        total_weight = sum(subclass_weights.values())

        if total_weight > 0:
            for subclass_id, weight in subclass_weights.items():
                if subclass_id in subclass_nt:
                    nt_type = subclass_nt[subclass_id]
                    if nt_type in nt_distribution:
                        nt_distribution[nt_type] += weight / total_weight
                    else:
                        nt_distribution["Other"] += weight / total_weight

        return nt_distribution

    def plot_marker_gene_analysis(self, region_pairs):
        """Plot D: Marker gene expression analysis for selected region pairs"""
        if region_pairs.empty or self.has_subclass_df is None or self.subclass_df is None:
            print("Warning: Missing data for marker gene analysis")
            return

        print("Plotting marker gene expression analysis...")

        # Extract genes from marker gene strings
        def extract_genes(gene_str):
            if pd.isna(gene_str) or gene_str == '':
                return []
            # Handle various possible separators
            genes = []
            for sep in [';', ',', '/', ' and ']:
                if sep in str(gene_str):
                    genes = [g.strip() for g in str(gene_str).split(sep)]
                    break
            if not genes and gene_str:
                genes = [str(gene_str).strip()]
            return [g for g in genes if g]

        # Process marker genes and transcription factors
        self.subclass_df['marker_genes'] = self.subclass_df['markers'].apply(extract_genes)
        self.subclass_df['tf_genes'] = self.subclass_df['transcription_factor_markers'].apply(extract_genes)

        # For each region pair, analyze marker gene differences
        for i, row in region_pairs.iterrows():
            region_pair = row['Region Pair']
            pair_type = row['Type']

            # Parse region pair
            region1_acronym, region2_acronym = region_pair.split('-')

            # Get region IDs
            region1_id = self.region_acronym_to_id.get(region1_acronym)
            region2_id = self.region_acronym_to_id.get(region2_acronym)

            if region1_id is None or region2_id is None:
                print(f"Warning: Cannot find region IDs: {region_pair}")
                continue

            # Get region subclass composition
            region1_subclasses = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region1_id]
            region2_subclasses = self.has_subclass_df[self.has_subclass_df[':START_ID(Region)'] == region2_id]

            if region1_subclasses.empty or region2_subclasses.empty:
                print(f"Warning: Cannot find region subclass data: {region_pair}")
                continue

            # Get important marker genes for each region
            region1_markers = self._get_region_markers(region1_subclasses)
            region2_markers = self._get_region_markers(region2_subclasses)

            # Create marker gene comparison plot
            self._plot_marker_gene_comparison(
                region1_acronym, region2_acronym,
                region1_markers, region2_markers,
                pair_type
            )

    def _get_region_markers(self, region_subclasses):
        """Get important marker genes for a region"""
        markers = defaultdict(float)
        tf_markers = defaultdict(float)

        # Get percentage of each subclass in the region
        total_weight = 0
        for _, row in region_subclasses.iterrows():
            subclass_id = row[':END_ID(Subclass)']
            weight = row['pct_cells:float']

            # Get subclass marker genes
            subclass_row = self.subclass_df[self.subclass_df['tran_id:ID(Subclass)'] == subclass_id]
            if not subclass_row.empty:
                # Regular marker genes
                for gene in subclass_row.iloc[0]['marker_genes']:
                    markers[gene] += weight

                # Transcription factor marker genes
                for gene in subclass_row.iloc[0]['tf_genes']:
                    tf_markers[gene] += weight

            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for gene in markers:
                markers[gene] /= total_weight

            for gene in tf_markers:
                tf_markers[gene] /= total_weight

        return {'markers': markers, 'tf_markers': tf_markers}

    def _plot_marker_gene_comparison(self, region1_name, region2_name, region1_markers, region2_markers, pair_type):
        """Plot marker gene comparison"""
        # Combine marker genes from both regions
        all_markers = set(list(region1_markers['markers'].keys()) + list(region2_markers['markers'].keys()))

        # Calculate differences
        marker_diff = []
        for gene in all_markers:
            r1_value = region1_markers['markers'].get(gene, 0)
            r2_value = region2_markers['markers'].get(gene, 0)
            if r1_value > 0 or r2_value > 0:  # Ignore genes not present in either region
                marker_diff.append({
                    'Gene': gene,
                    region1_name: r1_value,
                    region2_name: r2_value,
                    'Difference': r1_value - r2_value,
                    'Type': 'Regular Marker'
                })

        # Do the same for transcription factors
        all_tf_markers = set(list(region1_markers['tf_markers'].keys()) + list(region2_markers['tf_markers'].keys()))
        for gene in all_tf_markers:
            r1_value = region1_markers['tf_markers'].get(gene, 0)
            r2_value = region2_markers['tf_markers'].get(gene, 0)
            if r1_value > 0 or r2_value > 0:  # Ignore genes not present in either region
                marker_diff.append({
                    'Gene': gene,
                    region1_name: r1_value,
                    region2_name: r2_value,
                    'Difference': r1_value - r2_value,
                    'Type': 'Transcription Factor'
                })

        # Convert to DataFrame and sort
        diff_df = pd.DataFrame(marker_diff)

        # Remove genes with small differences, keep only meaningful differences
        diff_df = diff_df[abs(diff_df['Difference']) > 0.05]

        # Get genes with largest differences (both positive and negative)
        top_genes = diff_df.sort_values('Difference', ascending=False).head(10)
        bottom_genes = diff_df.sort_values('Difference', ascending=True).head(10)

        # Combine genes with significant differences
        plot_df = pd.concat([top_genes, bottom_genes])
        plot_df = plot_df.sort_values('Difference')

        # Plot gene difference chart
        plt.figure(figsize=(14, 10))

        # Create color mapping
        colors = []
        for _, row in plot_df.iterrows():
            if row['Difference'] < -0.05:
                colors.append(region_colors.get(region2_name, '#ff7f0e'))
            elif row['Difference'] > 0.05:
                colors.append(region_colors.get(region1_name, '#1f77b4'))
            else:
                colors.append('#808080')  # Gray for no significant difference

            # Add different shading for transcription factors
            if row['Type'] == 'Transcription Factor':
                # Make color darker
                colors[-1] = plt.cm.colors.to_rgba(colors[-1], alpha=0.7)

        # Plot horizontal bar chart
        bars = plt.barh(plot_df['Gene'] + ' (' + plot_df['Type'] + ')', plot_df['Difference'], color=colors)

        # Add reference lines
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=-0.05, color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.3)

        # Add labels and title
        plt.xlabel('Weighted Proportion Difference', fontsize=14)
        plt.ylabel('Marker Gene', fontsize=14)
        plt.title(f"{region1_name} vs {region2_name} Marker Gene Expression Differences", fontsize=16)

        # Add legend
        region1_patch = mpatches.Patch(color=region_colors.get(region1_name, '#1f77b4'),
                                       label=f'{region1_name} Enriched')
        region2_patch = mpatches.Patch(color=region_colors.get(region2_name, '#ff7f0e'),
                                       label=f'{region2_name} Enriched')
        tf_patch = mpatches.Patch(facecolor='gray', alpha=0.7, label='Transcription Factor')
        marker_patch = mpatches.Patch(facecolor='gray', alpha=1.0, label='Regular Marker')

        plt.legend(handles=[region1_patch, region2_patch, marker_patch, tf_patch], loc='lower right')

        # Adjust layout
        plt.tight_layout()

        # Save image
        output_file = self.output_dir / f"marker_gene_analysis_{region1_name}_vs_{region2_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved marker gene analysis to: {output_file}")
        plt.close()


def main():
    """Main function"""
    # Set data directory
    data_dir = Path("./knowledge_graph_v5")

    # Initialize analyzer
    analyzer = RegionAnalyzer(data_dir)

    # Calculate distance matrices
    if analyzer.calculate_distance_matrices():
        # Plot A: Distance relationship scatter plot and identify special region pairs
        special_pairs = analyzer.plot_distance_relationship()

        # Plot detailed analysis charts for special region pairs
        if special_pairs is not None and not special_pairs.empty:
            # B: Morphology feature radar charts
            analyzer.plot_morphology_radar(special_pairs)

            # C: Subclass composition bar charts
            analyzer.plot_subclass_composition(special_pairs)

            # D: Marker gene analysis (functional aspect)
            analyzer.plot_marker_gene_analysis(special_pairs)

            # Additional: Neurotransmitter distribution analysis
            analyzer.plot_neurotransmitter_distribution(special_pairs)

        print("Analysis complete! All charts saved to:", analyzer.output_dir)
    else:
        print("Error: Cannot calculate distance matrices, analysis terminated")


if __name__ == "__main__":
    main()