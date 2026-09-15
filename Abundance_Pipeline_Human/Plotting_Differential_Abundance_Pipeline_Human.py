"""
Visualization script (3x6 Grid of Cumulative Trajectory Line Graphs)
for A9 scCODA abundance data across ADNC categories.
Reads from TET1, TET2, and TET3 folders.
Groups cell types into Excitatory Neurons, Inhibitory Neurons, and Non-Neurons.
scCODA models were recorded as .csv files by the Differential_Abundance_Pipeline_GabittoDatabase pipeline.
     
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------
# Paths
RESULTS_BASE_DIR = Path(r"C:\Users\Fifiu\sead_project\A9\results")
TET_FOLDERS = {
    "TET1": RESULTS_BASE_DIR / "scCODA_A9_TET1_clinical_separate_comparisons",
    "TET2": RESULTS_BASE_DIR / "scCODA_A9_TET2_clinical_separate_comparisons",
    "TET3": RESULTS_BASE_DIR / "scCODA_A9_TET3_clinical_separate_comparisons"
}

VISUALS_DIR = RESULTS_BASE_DIR / "A9_advanced_visualizations_TET_unified"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# ADNC Progression Order
ADNC_ORDER = ['Not AD', 'Low', 'Intermediate', 'High']
TEST_CATEGORIES = ['Low', 'Intermediate', 'High']

# Cell Class Groupings requested
CELL_GROUPS = {
    "Excitatory Neurons": ["L5/6 NP", "L6 IT Car3", "L6b", "L6 CT", "L5 ET", "L5 IT", "L4 IT", "L6 IT", "L2/3 IT"],
    "Inhibitory Neurons": ["Chandelier", "Pvalb", "Sst", "Sst Chodl", "Vip", "Sncg", "Pax6", "Lamp5", "Lamp5 Lhx6"],
    "Non-Neurons": ["Microglia-PVM", "Microglia", "VLMC", "Endothelial", "Oligodendrocyte", "OPC", "Astrocyte", "Unknown"]
}

# Grid definitions
STATUS_COLUMNS = ["TET1+", "TET1-", "TET2+", "TET2-", "TET3+", "TET3-"]
GROUP_ROWS = ["Excitatory Neurons", "Inhibitory Neurons", "Non-Neurons"]

# ----------------------------------------------------------------------
# 1. HELPER FUNCTIONS & DATA PREP
# ----------------------------------------------------------------------
def split_cell_type(cell_type, tet_gene):
    """Split combined label into base cell type and TET status specific to the gene."""
    pos_suffix = f"_{tet_gene}+"
    neg_suffix = f"_{tet_gene}-"
    if pos_suffix in cell_type: 
        return cell_type.replace(pos_suffix, ''), f"{tet_gene}+"
    elif neg_suffix in cell_type: 
        return cell_type.replace(neg_suffix, ''), f"{tet_gene}-"
    else: 
        return cell_type, 'All'

def get_broad_class(cell_type):
    for group, types in CELL_GROUPS.items():
        if cell_type in types or any(t in cell_type for t in types): 
            return group
    return "Non-Neurons" # Fallback

def load_and_prepare_data():
    """
    Cycles through TET1, TET2, and TET3 folders, compiles CSVs, 
    extracts the respective TET status, and formats line chart data.
    """
    line_rows = []

    for tet_gene, folder in TET_FOLDERS.items():
        master_csv = folder / "all_comparisons_summary.csv"
        
        if not master_csv.exists():
            print(f"Warning: CSV not found at {master_csv}")
            continue
            
        df = pd.read_csv(master_csv)

        # Filter only ADNC comparisons if clinical_var exists
        if 'clinical_var' in df.columns:
            df = df[df['clinical_var'] == 'ADNC'].copy()

        # Apply TET Split Logic specific to the current gene
        df[['cell_type_base', 'tet_status']] = df['cell_type'].apply(lambda x: pd.Series(split_cell_type(x, tet_gene)))
        df['cell_class'] = df['cell_type_base'].apply(get_broad_class)

        # Determine significance column
        sig_col = 'credible_majority' if 'credible_majority' in df.columns else 'is_credible'
        
        # Apply strict significance threshold
        df['strict_sig'] = df[sig_col]
        
        # Prepare Line Chart Data (Cumulative Trajectory)
        for status in [f"{tet_gene}+", f"{tet_gene}-"]:
            sub_df = df[df['tet_status'] == status]
            if sub_df.empty: continue
                
            unique_cells = sub_df['cell_type_base'].unique()
            
            for ct in unique_cells:
                # Add baseline (0 shift for Not AD)
                line_rows.append({
                    'cell_type_base': ct,
                    'tet_status': status,
                    'cell_class': get_broad_class(ct),
                    'ADNC': 'Not AD',
                    'log2fc_mean': 0.0,
                    'strict_sig': False
                })
                
                # Fetch shifts for Low, Intermediate, High
                ct_data = sub_df[sub_df['cell_type_base'] == ct].set_index('test_category')
                for cat in TEST_CATEGORIES:
                    if cat in ct_data.index:
                        line_rows.append({
                            'cell_type_base': ct,
                            'tet_status': status,
                            'cell_class': get_broad_class(ct),
                            'ADNC': cat,
                            'log2fc_mean': ct_data.loc[cat, 'log2fc_mean'],
                            'strict_sig': ct_data.loc[cat, 'strict_sig']
                        })
                    else:
                        # Fill missing
                        line_rows.append({
                            'cell_type_base': ct, 'tet_status': status, 'cell_class': get_broad_class(ct),
                            'ADNC': cat, 'log2fc_mean': 0.0, 'strict_sig': False
                        })
                        
    line_df = pd.DataFrame(line_rows)
    return line_df

# ----------------------------------------------------------------------
# 2. PLOT: UNIFIED CUMULATIVE TRAJECTORY LINE GRAPHS
# ----------------------------------------------------------------------
def plot_unified_cumulative_lines(line_df):
    print("\nGenerating Unified Cumulative Trajectory Line Graph (3x6 Grid)...")

    # Slightly wider figsize (24 instead of 20), still keeping tall framing per subplot
    fig, axes = plt.subplots(3, 6, figsize=(24, 24), sharex=True)
    # Intermediate suptitle font size
    #fig.suptitle("A9 - Cumulative log2FC Abundance Trajectory Across ADNC Stages (All TETs)", fontsize=26, y=0.98)

    for r_idx, group in enumerate(GROUP_ROWS):
        for c_idx, status in enumerate(STATUS_COLUMNS):
            ax = axes[r_idx, c_idx]
            
            group_data = line_df[(line_df['cell_class'] == group) & (line_df['tet_status'] == status)].copy()
            
            if group_data.empty:
                ax.set_visible(False)
                continue

            group_data['ADNC'] = pd.Categorical(group_data['ADNC'], categories=ADNC_ORDER, ordered=True)
            group_data = group_data.sort_values(['cell_type_base', 'ADNC'])
            unique_cts = sorted(group_data['cell_type_base'].unique())
            
            palette = sns.color_palette("husl", len(unique_cts))
            color_map = dict(zip(unique_cts, palette))

            # Plot lines manually to control transparency per cell type
            for ct in unique_cts:
                ct_data = group_data[group_data['cell_type_base'] == ct]
                col = color_map[ct]
                
                # Check if this cell type has ANY significant transitions
                has_sig = ct_data['strict_sig'].any()
                line_alpha = 1.0 if has_sig else 0.3
                
                # Plot the line with determined transparency
                sns.lineplot(data=ct_data, x='ADNC', y='log2fc_mean', 
                             color=col, marker='o', ax=ax, sort=False, 
                             alpha=line_alpha, label=ct)
                
                # Overlay asterisks for significant transitions relative to baseline
                for _, row in ct_data.iterrows():
                    if row['strict_sig']:
                        ax.annotate('*', 
                                    xy=(row['ADNC'], row['log2fc_mean']), 
                                    xytext=(0, 4), # Offset vertically above point
                                    textcoords='offset points',
                                    ha='center', va='bottom',
                                    color=col,
                                    fontsize=18, fontweight='bold') # Intermediate asterisk

            ax.axhline(0, color='black', linestyle='--', alpha=0.5)

            # Set requested y-axis ranges per group
            if group == "Excitatory Neurons":
                ax.set_ylim(-0.6, 0.6)
            elif group == "Inhibitory Neurons":
                ax.set_ylim(-0.5, 0.5)
            elif group == "Non-Neurons":
                ax.set_ylim(-0.5, 0.5)
            
            # Formatting labels and titles with intermediate font sizes
            if r_idx == 0:
                ax.set_title(status, fontsize=25, fontweight='bold', pad=15)
            
            if c_idx == 0:
                ax.set_ylabel(f"Cumulative log2FC", fontsize=22)
            else:
                ax.set_ylabel("")

            if r_idx == 2:
                ax.set_xlabel("ADNC Stage", fontsize=22)
                # Correct rotation displacement using horizontal alignment (ha) and anchor
                ax.set_xticks(range(len(ADNC_ORDER)))
                ax.set_xticklabels(ADNC_ORDER, rotation=45, ha='right', rotation_mode='anchor', fontsize=20)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)
                
            ax.tick_params(axis='y', labelsize=17)
            
            # 1) Restore legend to all graphs
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=14)
            
            # 2) Force the lines inside the legend box to be fully opaque
            # This accounts for compatibility across different matplotlib versions
            legend_lines = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
            for line in legend_lines:
                line.set_alpha(1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to leave space for suptitle
    
    out_file = VISUALS_DIR / "Unified_A9_CumulativeLines_ADNC_AllTETs.pdf"
    # Save as publication quality 300 dpi
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved Unified Cumulative Trajectory Grid to {out_file} at 300 DPI")

# ----------------------------------------------------------------------
# 3. MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    print("="*70)
    print("STARTING UNIFIED A9 TET1/TET2/TET3 VISUALIZATION PIPELINE")
    print("="*70)
    
    line_df = load_and_prepare_data()
    
    if line_df is not None and not line_df.empty:
        plot_unified_cumulative_lines(line_df)
    else:
        print("\n❌ Could not load valid TET data. Please verify your RESULTS_BASE_DIR and contents.")

    print("\n" + "="*70)
    print(f"ALL DONE. Visualizations saved to: {VISUALS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()