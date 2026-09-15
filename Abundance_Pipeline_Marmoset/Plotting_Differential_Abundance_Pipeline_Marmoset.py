"""
3x6 Side-by-Side Trajectory Matrix for Marmoset TET Enzyme Family (PFC).
Rows: Cell Classes (Neuronal, Glial, Other)
Columns: Enzyme Status Pairs (TET1+, TET1-, TET2+, TET2-, TET3+, TET3-)
scCODA models were recorded as .csv files (e.g. scCODA_results_PFC_tet1_split)
by the Differential_Abundance_Pipeline_SchroederDatabase pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 0. CONFIGURATION & MACHINE PATHS
# ----------------------------------------------------------------------
REGION = "PFC"

FILE_PATHS = {
    "TET1": Path(f"results/scCODA_TET1_refcycled_{REGION}/scCODA_results_{REGION}_tet1_split.csv"),
    "TET2": Path(f"results/scCODA_TET2_refcycled_{REGION}/scCODA_results_{REGION}_tet2_split.csv"),
    "TET3": Path(f"results/scCODA_TET3_refcycled_{REGION}/scCODA_results_{REGION}_tet3_split.csv")
}

RESULTS_DIR = Path(f"results/TET_side_by_side_visualizations_{REGION}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHRONO_MODELS = [
    "model1_GD135_vs_neonate",
    "model2_neonate_vs_7months",
    "model3_7months_vs_14months",
    "model4_14months_vs_30months",
    "model5_30months_vs_aged"
]

AGE_ORDER = ["GD135", "neonate", "7months", "14months", "30months", "aged"]
X_DISPLAY_LABELS = ["GD 135", "Neonate", "7 months", "14 months", "30 months", "Aged"]

CELL_GROUPS = {
    "Neuronal": ["Neuron Excitatory", "Neuron Inhibitory", "Neuron Mixed"],
    "Glial": ["Astrocyte", "Microglia", "Oligodendrocyte", "OPC", "Ependymal"],
    "Other": ["Endothelial", "Fibroblast", "Mural", "ChoroidPlexus", "Other", "Unknown"]
}

CELL_COLORS = {
    "Neuron Excitatory": "#1e3a8a",   # Royal Deep Blue
    "Neuron Inhibitory": "#b91c1c",   # Crimson Red
    "Neuron Mixed":      "#6d28d9",   # Rich Amethyst Purple
    "Astrocyte":         "#047857",   # Emerald Green
    "Microglia":         "#ea580c",   # Vibrant Dark Orange
    "Oligodendrocyte":   "#0369a1",   # Deep Ocean Blue
    "OPC":               "#be185d",   # Deep Magenta Pink
    "Ependymal":         "#4b5563",   # Cool Slate Grey
    "Endothelial":       "#111827",   # Jet Black
    "Fibroblast":        "#78350f",   # Warm Amber Brown
    "Mural":             "#9d174d",   # Deep Wine
    "ChoroidPlexus":     "#0d9488",   # Teal Green
    "Other":             "#6b7280",   # Neutral Grey
    "Unknown":           "#9ca3af"    # Light Grey
}

def get_broad_class(cell_type):
    for group, types in CELL_GROUPS.items():
        if cell_type in types: return group
    return "Other"

# ----------------------------------------------------------------------
# 2. DATA PROCESSING PIPELINE
# ----------------------------------------------------------------------
def format_sccoda_data(raw_df, enzyme_name):
    df = raw_df.copy()
    if 'log2-fold change' in df.columns:
        df.rename(columns={'log2-fold change': 'log2fc_mean'}, inplace=True)
        
    # Translate raw cell type labels to clean publication formatting
    df['cell_type_base'] = df['cell_type_base'].replace({
        'NeuronExcit': 'Neuron Excitatory',
        'NeuronInh': 'Neuron Inhibitory',
        'NeuronInhib': 'Neuron Inhibitory', # Fallback just in case
        'NeuronMixed': 'Neuron Mixed'
    })
        
    status_col = f"{enzyme_name.lower()}_status"
    if status_col in df.columns:
        df['tet_status'] = df[status_col].apply(lambda x: '+' if '+' in str(x) else '-')
    else:
        possible_status_cols = ['Cell Type', 'tet3_status', 'tet2_status', 'tet1_status']
        found = False
        for col in possible_status_cols:
            if col in df.columns:
                df['tet_status'] = df[col].apply(lambda x: '+' if '+' in str(x) else '-')
                found = True
                break
        if not found:
            df['tet_status'] = '+'

    model_mapping = {
        ('GD135', 'neonate'): 'model1_GD135_vs_neonate',
        ('neonate', '7months'): 'model2_neonate_vs_7months',
        ('7months', '14months'): 'model3_7months_vs_14months',
        ('14months', '30months'): 'model4_14months_vs_30months',
        ('30months', 'aged'): 'model5_30months_vs_aged'
    }
    
    df['model'] = df.apply(lambda r: model_mapping.get((r['baseline_group'], r['comparison_group'])), axis=1)
    df = df.dropna(subset=['model']).copy()
    df['enzyme'] = enzyme_name
    return df

def prepare_combined_trajectories(all_dfs):
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['strict_sig'] = combined_df['is_credible']
    
    chrono_df = combined_df[combined_df['model'].isin(CHRONO_MODELS)].copy()
    cumulative_rows = []
    
    for enzyme in ["TET1", "TET2", "TET3"]:
        for status in ['+', '-']:
            sub = chrono_df[(chrono_df['enzyme'] == enzyme) & (chrono_df['tet_status'] == status) & (chrono_df['region'] == REGION)]
            if sub.empty: continue
                
            unique_cts = sub['cell_type_base'].unique()
            grid = pd.MultiIndex.from_product([CHRONO_MODELS, unique_cts], names=['model', 'cell_type_base']).to_frame(index=False)
            grid['region'] = REGION
            grid['tet_status'] = status
            grid['enzyme'] = enzyme
            
            merged = pd.merge(grid, sub, on=['model', 'cell_type_base', 'region', 'tet_status', 'enzyme'], how='left')
            merged['log2fc_mean'] = merged['log2fc_mean'].fillna(0)
            merged['strict_sig'] = merged['strict_sig'].fillna(False)
            
            for ct in unique_cts:
                ct_data = merged[merged['cell_type_base'] == ct].set_index('model').reindex(CHRONO_MODELS)
                deltas = ct_data['log2fc_mean'].tolist()
                sigs = ct_data['strict_sig'].tolist()
                
                cum_vals = [0] + list(np.cumsum(deltas))
                transition_sigs = [False] + sigs
                
                for i, age in enumerate(AGE_ORDER):
                    cumulative_rows.append({
                        'region': REGION, 'enzyme': enzyme, 'tet_status': status,
                        'cell_type_base': ct, 'cell_class': get_broad_class(ct), 'age': age,
                        'cumulative_log2fc': cum_vals[i], 'transition_sig': transition_sigs[i]
                    })
                        
    return pd.DataFrame(cumulative_rows)

# ----------------------------------------------------------------------
# 3. SIDE-BY-SIDE MATRIX PLOTTING
# ----------------------------------------------------------------------
def plot_side_by_side_matrix(cum_df):
    print(f"Generating 3x6 side-by-side layout matrix for {REGION}...")
    
    # Total Figure Dimensions Matching Request
    fig, axes = plt.subplots(3, 6, figsize=(24, 24), sharex=True)
    
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Font Sizes: Main Figure Title (Suptitle)
    #fig.suptitle(f"{REGION} - Cumulative log2FC Lifespan Abundance Trajectories of TET Enzyme Families", 
                 #fontsize=26, fontweight='bold', y=0.98)
    
    cell_classes = ["Neuronal", "Glial", "Other"]
    
    columns_config = [
        {"enzyme": "TET1", "status": "+", "label": "TET1+"},
        {"enzyme": "TET1", "status": "-", "label": "TET1-"},
        {"enzyme": "TET2", "status": "+", "label": "TET2+"},
        {"enzyme": "TET2", "status": "-", "label": "TET2-"},
        {"enzyme": "TET3", "status": "+", "label": "TET3+"},
        {"enzyme": "TET3", "status": "-", "label": "TET3-"}
    ]
    
    y_limits = {
        "Neuronal": (-1.0, 0.8),
        "Glial":    (-1.0, 3.5),
        "Other":    (-0.5, 1.0)
    }
    
    for r_idx, group in enumerate(cell_classes):
        for c_idx, config in enumerate(columns_config):
            ax = axes[r_idx, c_idx]
            
            # Structural Adjustment: Explicit Removal of Background Gray Lines
            ax.grid(False)
            
            enzyme = config["enzyme"]
            status = config["status"]
            
            subset = cum_df[(cum_df['enzyme'] == enzyme) & 
                            (cum_df['tet_status'] == status) & 
                            (cum_df['cell_class'] == group)].copy()
                            
            if subset.empty:
                ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1.0)
                continue
                
            subset['age'] = pd.Categorical(subset['age'], categories=AGE_ORDER, ordered=True)
            subset = subset.sort_values(['cell_type_base', 'age'])
            group_cts = sorted(subset['cell_type_base'].unique())
            
            # Formulate Clean Legend Handles
            type_handles = []
            for ct in group_cts:
                line_color = CELL_COLORS.get(ct, CELL_COLORS["Other"])
                type_handles.append(Line2D([0], [0], color=line_color, marker='o', markersize=5.0, 
                                           linestyle='-', linewidth=1.5, label=ct))
            
            # Font Sizes: Legend Text Layout
            leg = ax.legend(handles=type_handles, loc='upper left', frameon=True, 
                            facecolor='white', framealpha=0.85, fontsize=13)
            
            # Styling Elements: Legend Opacity Override Control
            if leg:
                for leg_handle in leg.legend_handles:
                    leg_handle.set_alpha(1.0)
            
            # Plot individual cell trajectories
            for ct in group_cts:
                line_color = CELL_COLORS.get(ct, CELL_COLORS["Other"])
                ct_data = subset[subset['cell_type_base'] == ct]
                
                has_any_sig = ct_data['transition_sig'].any()
                
                # Styling Elements: Line Transparency Alpha Rules
                l_alpha = 1.0 if has_any_sig else 0.3
                l_width = 1.6 if has_any_sig else 1.0
                m_size  = 5.5 if has_any_sig else 3.5
                
                ax.plot(X_DISPLAY_LABELS, ct_data['cumulative_log2fc'], 
                        marker='o', linestyle='-', linewidth=l_width, markersize=m_size, 
                        color=line_color, alpha=l_alpha)
                
                # Draw significance labels
                for _, row in ct_data.iterrows():
                    if row['transition_sig']:
                        y_offset = 4
                        if enzyme == "TET3" and ct == "Astrocyte" and status == "+" and row['age'] in ["30months", "aged"]:
                            y_offset = 0.5
                            
                        # Font Sizes: Significance Asterisks (*) Adjustments
                        ax.annotate("*", xy=(X_DISPLAY_LABELS[AGE_ORDER.index(row['age'])], row['cumulative_log2fc']), 
                                    xytext=(0, y_offset), textcoords='offset points',
                                    ha='center', va='bottom', color=line_color,
                                    fontsize=18, fontweight='bold', alpha=l_alpha)
            
            # Axis ranges and tick marks controls
            ax.set_ylim(y_limits[group])
            ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1.0)
            
            # Font Sizes: Tick Marks (X and Y axes)
            ax.tick_params(axis='both', labelsize=14)
            
            # Font Sizes: Subplot Titles (Top Row Only)
            if r_idx == 0:
                ax.set_title(config["label"], fontsize=20, fontweight='bold', pad=12)
                
            # Font Sizes: Axis Titles / Labels (Left Column Only)
            #if c_idx == 0:
                #ax.set_ylabel(f"{group} cells\n\nCumulative log2FC", fontsize=16, fontweight='bold')
            #else:
                #ax.set_ylabel("")
                
            # Font Sizes: Axis Titles / Labels (Bottom Row Only)
            if r_idx == 2:
                ax.set_xlabel("Lifespan Stage", fontsize=16, fontweight='bold')
                # Styling Elements: X-Axis Label Anchored Rotation
                ax.set_xticklabels(X_DISPLAY_LABELS, rotation=45, ha='right', rotation_mode='anchor')
            else:
                ax.set_xlabel("")
                
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    
    # Styling Elements: DPI Publication Quality High Resolution Output
    save_path = RESULTS_DIR / f"Plot_SideBySide_TrajectoryMatrix_{REGION}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"--> Matrix plot saved successfully at:\n    {save_path}")
    plt.close()

# ----------------------------------------------------------------------
# 4. PIPELINE CONTROL EXECUTION
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print("======================================================================")
    print(f"STARTING SIDE-BY-SIDE TRAJECTORY MATRIX PIPELINE FOR: {REGION}")
    print("======================================================================")
    
    formatted_dfs = []
    for enzyme, path in FILE_PATHS.items():
        if path.exists():
            print(f"Loading data matrix file for {enzyme}...")
            raw = pd.read_csv(path)
            formatted = format_sccoda_data(raw, enzyme)
            formatted_dfs.append(formatted)
        else:
            print(f"❌ Warning: Cannot locate source file for {enzyme} at {path}")
            
    if len(formatted_dfs) == 3:
        trajectory_df = prepare_combined_trajectories(formatted_dfs)
        plot_side_by_side_matrix(trajectory_df)
        print("\nTrajectory matrix visualization successfully completed.")
    else:
        print("\n❌ Error: Pipeline aborted. Verify that all 3 file inputs exist.")