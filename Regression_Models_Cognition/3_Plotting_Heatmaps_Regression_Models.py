"""
Heatmaps of multivariate regression models to evaluate associations of cell percentages with cognitive scores. Based
on the Seattle Alzheimer's Disease Brain Cell Atlas(https://brain-map.org/consortia/sea-ad) - dataset of Gabitto et al. 2024 (https://doi.org/10.1038/s41593-024-01774-5) and Mukherjee et al. 2020 (https://doi.org/10.1038/s41380-018-0298-8).

Before using this script, the user need to run the "Regression_Models_Cognition.py" script to generate the "TET_cell_subclass_regression_results.csv" file.  

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
RESULTS_CSV = "TET_cell_subclass_regression_results.csv"
OUTPUT_DIR = "Plots_Unified_Heatmaps"

P_VAL_THRESHOLD = 0.05

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def categorize_cell_type(cell_name):
  """Categorizes SEA-AD / Allen Brain cell types into 3 main groups."""
  ex_keywords = ['IT', 'ET', 'NP', 'CT', 'L6b', 'Excitatory']
  inh_keywords = [
      'Pvalb',
      'SST',
      'VIP',
      'Lamp5',
      'Sncg',
      'Chandelier',
      'Pax6',
      'Inhibitory',
  ]
  non_keywords = [
      'Astrocyte',
      'Oligodendrocyte',
      'OPC',
      'Microglia',
      'Microglia-PVM',
      'Endothelial',
      'VLMC',
      'Pericyte',
      'Glia',
      'Fibroblast',
  ]

  cell_upper = str(cell_name).upper()

  for kw in ex_keywords:
    if kw.upper() in cell_upper:
      return 'Excitatory neurons'
  for kw in inh_keywords:
    if kw.upper() in cell_upper:
      return 'Inhibitory neurons'
  for kw in non_keywords:
    if kw.upper() in cell_upper:
      return 'Non-neurons'

  return 'Other'


def plot_unified_heatmap(
    metric_subset, metric_label, cog_tests, all_cells, vmax, p_val_col
):
  """Generates a single unified heatmap."""

  # Categorize and sort cells to structure the Y-axis divisions
  cell_categories = {cell: categorize_cell_type(cell) for cell in all_cells}
  cat_order = [
      'Excitatory neurons',
      'Inhibitory neurons',
      'Non-neurons',
      'Other',
  ]

  ordered_cells = []
  group_boundaries = []
  current_y = 0

  for cat in cat_order:
    cells_in_cat = sorted(
        [c for c, cat_val in cell_categories.items() if cat_val == cat]
    )
    if cells_in_cat:
      ordered_cells.extend(cells_in_cat)
      current_y += len(cells_in_cat)
      group_boundaries.append((current_y, cat))

  if group_boundaries:
    group_boundaries.pop()  # Remove bottom-most boundary

  # Split into Normal Cognition and Alzheimer's Disease
  df_nc = metric_subset[
      ~metric_subset['Covariate'].str.contains('Calculated AD Slope')
  ]
  df_ad = metric_subset[
      metric_subset['Covariate'].str.contains('Calculated AD Slope')
  ]

  # Create multi-columns (TET1 -> 5 tests, TET2 -> 5 tests, TET3 -> 5 tests)
  enzymes = ['TET1', 'TET2', 'TET3']
  multi_cols = [(tet, test) for tet in enzymes for test in cog_tests]
  x_tick_labels = [test for _, test in multi_cols]

  # Dynamically calculate figure size
  fig_width = 22
  fig_height = max(8, len(ordered_cells) * 0.4)

  # 3 columns: left group, right group, dedicated colorbar axis
  fig, axes = plt.subplots(
      nrows=1,
      ncols=3,
      figsize=(fig_width, fig_height),
      gridspec_kw={'wspace': 0.05, 'width_ratios': [1, 1, 0.02]},
  )

  groups_data = [
      ('Cognitively Unimpaired', df_nc, axes[0], False),
      ("Alzheimer's Disease", df_ad, axes[1], True),
  ]

  for group_name, grp_df, ax, show_cbar in groups_data:
    # Build Matrices
    coef_matrix = pd.DataFrame(
        np.nan,
        index=ordered_cells,
        columns=pd.MultiIndex.from_tuples(multi_cols),
    )
    pval_matrix = pd.DataFrame(
        np.nan,
        index=ordered_cells,
        columns=pd.MultiIndex.from_tuples(multi_cols),
    )

    for cell in ordered_cells:
      for tet, test in multi_cols:
        match = grp_df[
            (grp_df['Cell_Subclass'] == cell)
            & (grp_df['Cognitive_Score'] == test)
            & (grp_df['Enzyme'] == tet)
        ]
        if not match.empty:
          coef_matrix.loc[cell, (tet, test)] = match['Coefficient'].values[0]
          pval_matrix.loc[cell, (tet, test)] = match[p_val_col].values[0]

    # Asterisk Annotation Matrix
    annot_matrix = np.full(coef_matrix.shape, '', dtype=object)
    for r in range(coef_matrix.shape[0]):
      for c in range(coef_matrix.shape[1]):
        p_val = pval_matrix.iloc[r, c]
        if pd.notna(p_val):
          if p_val < 0.001:
            annot_matrix[r, c] = '***'
          elif p_val < 0.01:
            annot_matrix[r, c] = '**'
          elif p_val < 0.05:
            annot_matrix[r, c] = '*'

    # Fill NaNs with 0 for aesthetic blank squares
    coef_matrix = coef_matrix.fillna(0)

    # Set dedicated colorbar axis and properties
    cbar_ax = axes[2] if show_cbar else None
    cbar_kws = {'label': 'Beta Coefficient', 'pad': 0.03} if show_cbar else None

    sns.heatmap(
        coef_matrix,
        ax=ax,
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        center=0,
        annot=annot_matrix,
        fmt='',
        annot_kws={
            'fontsize': 13,
            'color': 'black',
            'ha': 'center',
            'va': 'center',
        },
        linewidths=1.0,
        linecolor='white',
        square=False,
        cbar=show_cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        yticklabels=ordered_cells if ax == axes[0] else False,
    )

    # Format X-axis ticks (Move to top)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(multi_cols)) + 0.5)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='left', fontsize=12)
    ax.set_xlabel('')

    # Format Borders & Thick Division Lines
    for _, spine in ax.spines.items():
      spine.set_visible(True)
      spine.set_linewidth(2)
      spine.set_color('black')

    for y_pos, _ in group_boundaries:
      ax.axhline(y_pos, color='black', linewidth=2.5, zorder=5)

    num_tests = len(cog_tests)
    ax.axvline(num_tests, color='black', linewidth=2.5, zorder=5)
    ax.axvline(num_tests * 2, color='black', linewidth=2.5, zorder=5)

    # Y-axis labels
    if ax == axes[0]:
      ax.set_yticklabels(ordered_cells, rotation=0, fontsize=12)
      ax.set_ylabel('')

  safe_metric = metric_label.replace('%', 'pct').replace(' ', '_')
  filename = f'Unified_TET_Heatmap_{safe_metric}.pdf'
  plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
  plt.close()
  print(f'Saved Publication Figure: {filename}')


# ==========================================
# MAIN SCRIPT
# ==========================================
def load_clean_results(csv_path):
  """Locates the header row and auto-detects delimiter to prevent ParserError."""
  with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

  # Find line index where actual table header begins
  header_idx = None
  for idx, line in enumerate(lines):
    if 'Cell_Subclass' in line and 'Covariate' in line:
      header_idx = idx
      break

  if header_idx is None:
    header_idx = 0  # Default to top line if no metadata header text exists

  # Auto-detect separator
  header_line = lines[header_idx]
  sep = ';' if ';' in header_line and ',' not in header_line else ','

  return pd.read_csv(csv_path, skiprows=header_idx, sep=sep)


def main():
  if not os.path.exists(RESULTS_CSV):
    print(f'Error: Could not find results file: {RESULTS_CSV}')
    return

  print(f'Loading regression results from {RESULTS_CSV}...')
  df = load_clean_results(RESULTS_CSV)

  # Fallback logic to dynamically select the correct FDR column
  if 'Adjusted_P_Value_FDR' in df.columns:
    p_val_col = 'Adjusted_P_Value_FDR'
  elif 'Adjusted_P_Value (FDR)' in df.columns:
    p_val_col = 'Adjusted_P_Value (FDR)'
  else:
    p_val_col = 'Raw_P_Value'
    print(
        'Warning: Neither Adjusted_P_Value_FDR nor Adjusted_P_Value (FDR)'
        ' found. Falling back to Raw_P_Value.'
    )

  # Standardize names of cognitive domains
  df['Cognitive_Score'] = df['Cognitive_Score'].replace({
      'MEM_E': 'MEM',
      'EXF_E': 'EXF',
      'LAN_E': 'LAN',
      'VSP_E': 'VSP',
  })

  # Filter out CASI to keep only primary cognitive domains
  df = df[df['Cognitive_Score'] != 'CASI']

  df_plot = df[
      df['Covariate'].str.contains('Calculated AD Slope|TET', regex=True)
  ].copy()
  df_plot = df_plot[~df_plot['Covariate'].str.contains(':')]

  cog_tests = df_plot['Cognitive_Score'].unique()
  if len(cog_tests) > 5:
    cog_tests = cog_tests[:5]
  all_cells = df_plot['Cell_Subclass'].dropna().unique()

  all_coefs = df_plot['Coefficient'].dropna().values
  vmax = np.max(np.abs(all_coefs)) * 1.05 if len(all_coefs) > 0 else 1.0

  metrics = df_plot['Predictor_Metric'].unique()

  for metric in metrics:
    print(f'\nProcessing matrices for metric: {metric}')
    metric_subset = df_plot[df_plot['Predictor_Metric'] == metric]

    print('  -> Generating Unified Heatmap...')
    plot_unified_heatmap(
        metric_subset, metric, cog_tests, all_cells, vmax, p_val_col
    )

  print(
      f"\nAll correlation heatmaps generated successfully in '{OUTPUT_DIR}'!"
  )


if __name__ == '__main__':
  main()