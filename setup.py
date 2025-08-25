"""Setup and configuration for analysis."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances
from umap import UMAP
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from upsetplot import from_memberships, UpSet
import warnings
import nrrd

# Set up logging
from loguru import logger

logger.add("analysis.log", rotation="10 MB")

# Plotting style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Custom color maps
morph_cmap = sns.color_palette("rocket", as_cmap=True)
module_cmap = sns.color_palette("mako", as_cmap=True)
region_cmap = sns.color_palette("viridis", as_cmap=True)

# Paths
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("./output/figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Constants
RANDOM_STATE = 42
FDR_THRESHOLD = 0.05

# The 9 morphological features of interest
MORPH_FEATURES = [
    "axonal_length",  # 轴突总长
    "axonal_branches",  # 轴突分叉数
    "axonal_bifurcation_remote_angle",  # 远端分叉角
    "axonal_maximum_branch_order",  # 最高分叉阶
    "dendritic_length",  # 树突总长
    "dendritic_branches",  # 树突分叉数
    "dendritic_bifurcation_remote_angle",  # 树突远端角
    "dendritic_maximum_branch_order",  # 树突最大阶
]

# Feature name mapping for better display
MORPH_FEATURE_NAMES = {
    "axonal_length": "Total Length",
    "axonal_branches": "Number of Bifurcations",
    "axonal_bifurcation_remote_angle": "Average Bifurcation Angle Remote",
    "axonal_maximum_branch_order": "Max Branch Order",
    "dendritic_length": "Total Length",
    "dendritic_branches": "Number of Bifurcations",
    "dendritic_bifurcation_remote_angle": "Average Bifurcation Angle Remote",
    "dendritic_maximum_branch_order": "Max Branch Order",
}

# Gene modules - these would be defined based on Allen MERFISH data
# We'll define placeholder module definitions and populate them later
GENE_MODULES = {
    # Cell-type markers (10 groups)
    "IT_Neurons": [],  # Intratelencephalic projection neurons
    "ET_Neurons": [],  # Extratelencephalic projection neurons
    "CT_Neurons": [],  # Corticothalamic projection neurons
    "PV_Interneurons": [],  # Parvalbumin interneurons
    "SST_Interneurons": [],  # Somatostatin interneurons
    "VIP_Interneurons": [],  # Vasoactive intestinal peptide interneurons
    "Lamp5_Interneurons": [],  # Lamp5 interneurons
    "NPY_Interneurons": [],  # Neuropeptide Y interneurons
    "Astrocytes": [],  # Astrocytes
    "Oligodendrocytes": [],  # Oligodendrocytes

    # GO/KEGG functional modules (8 groups)
    "Axon_Guidance": [],  # Axon guidance
    "Cytoskeleton": [],  # Cytoskeleton
    "Myelination": [],  # Myelination
    "Synaptic_Transmission": [],  # Synaptic transmission
    "Energy_Metabolism": [],  # Energy metabolism
    "Protein_Transport": [],  # Protein transport
    "Ion_Transport": [],  # Ion transport
    "Transcription": []  # Transcription
}

# Define module types for grouping in visualization
MODULE_TYPES = {
    "IT_Neurons": "Cell_Type",
    "ET_Neurons": "Cell_Type",
    "CT_Neurons": "Cell_Type",
    "PV_Interneurons": "Cell_Type",
    "SST_Interneurons": "Cell_Type",
    "VIP_Interneurons": "Cell_Type",
    "Lamp5_Interneurons": "Cell_Type",
    "NPY_Interneurons": "Cell_Type",
    "Astrocytes": "Cell_Type",
    "Oligodendrocytes": "Cell_Type",

    "Axon_Guidance": "Function",
    "Cytoskeleton": "Function",
    "Myelination": "Function",
    "Synaptic_Transmission": "Function",
    "Energy_Metabolism": "Function",
    "Protein_Transport": "Function",
    "Ion_Transport": "Function",
    "Transcription": "Function"
}