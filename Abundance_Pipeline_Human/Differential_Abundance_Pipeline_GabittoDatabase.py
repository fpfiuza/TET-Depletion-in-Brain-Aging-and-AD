"""
Comprehensive scCODA analysis of cell proportions in humans, 
based on the Seattle Alzheimer's Disease Brain Cell Atlas(https://brain-map.org/consortia/sea-ad) - dataset of Gabitto et al. 2024 (https://doi.org/10.1038/s41593-024-01774-5).
Evaluates the TET-split(or any gene-split) abundance for a single brain region. In this case, TET3 is being evaluated in the A9 brain region.
The user should first download the .h5ad files (MTG and A9) available at https://registry.opendata.aws/allen-sea-ad-atlas/
This script runs each .h5ad file systematically shifting the baseline 
reference to extract every possible pairwise ADNC comparison.
It utilizes Reference cycling (majority voting) for each baseline model.

MCMC parameters: 20000 iterations; 5000 burn-in

"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# scCODA imports
# ----------------------------------------------------------------------
from sccoda.util import comp_ana as mod
from sccoda.util import cell_composition_data as dat

# ----------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------
DATA_FILE = "dad4819b-4c14-439c-b32a-2c8d68bd22e1.h5ad"
RESULTS_DIR = Path("results/scCODA_A9_TET3_clinical_separate_comparisons")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
TET3_THRESHOLD = 0.5                      # expression > 0.5 → TET3+
MIN_CELLS_PER_TYPE = 50                   # minimum total cells for a (type,status) combination
MIN_DONORS_PER_TYPE = 2                   # minimum donors with non‑zero count
MCMC_ITER = 20000
FDR_LEVEL = 0.05

# Define ADNC comparisons (ALL vs "Not AD")
ADNC_REFERENCE = 'Not AD'
ADNC_TEST_CATEGORIES = ['Low', 'Intermediate', 'High']

# Custom order for cell types in forest plots
CUSTOM_CELL_ORDER = [
    "Microglia-PVM", "VLMC", "Endothelial", "Oligodendrocyte", "OPC",
    "Astrocyte", "L5/6 NP", "L6 IT Car3", "L6b", "L6 CT", "L5 ET", "L5 IT",
    "L4 IT", "L6 IT", "L2/3 IT", "Chandelier", "Pvalb", "Sst", "Sst Chodl",
    "Vip", "Sncg", "Pax6", "Lamp5", "Lamp5 Lhx6"
]

# Mapping for nicer cell type names
CELL_TYPE_RENAME = {
    "Microglia-PVM": "Microglia",
}

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
print("="*70)
print("LOADING ANNDATA - A9 REGION")
print("="*70)
print("Loading AnnData...")
adata = sc.read_h5ad(DATA_FILE, backed='r')
print(f"Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")

# ----------------------------------------------------------------------
# 2. LOCATE TET3 GENE
# ----------------------------------------------------------------------
possible_names = ["ENSG00000187605", "TET3", "Tet3"]
tet3_gene = next((name for name in possible_names if name in adata.var_names), None)

if tet3_gene is None:
    raise ValueError("TET3 gene not found in var_names.")
print(f"Using gene: {tet3_gene}")

chunk_size = 100000
n_cells = adata.n_obs
tet3_expr = np.zeros(n_cells, dtype=np.float32)

print(f"Extracting TET3 expression for {n_cells:,} cells in chunks...")
for i in range(0, n_cells, chunk_size):
    end = min(i + chunk_size, n_cells)
    chunk = adata[i:end, tet3_gene].X
    if hasattr(chunk, 'toarray'):
        tet3_expr[i:end] = chunk.toarray().flatten()
    elif hasattr(chunk, 'A'):
        tet3_expr[i:end] = chunk.A.flatten()
    else:
        tet3_expr[i:end] = chunk.flatten()
    print(f"   Processed cells {i:,} to {end:,}")

obs_df = adata.obs.copy()
adata.file.close()

obs_df['TET3_expr'] = tet3_expr

# ----------------------------------------------------------------------
# 3. DEFINE TET3 POSITIVITY & CELL TYPES
# ----------------------------------------------------------------------
obs_df["TET3_positive"] = (tet3_expr > TET3_THRESHOLD).astype(bool)
print(f"\nTET3+ cells: {obs_df['TET3_positive'].sum():,} ({obs_df['TET3_positive'].mean()*100:.2f}%)")

obs_df["cell_type_subclass_tet3"] = (
    obs_df["Subclass"].astype(str) + "_" + obs_df["TET3_positive"].map({True: "TET3+", False: "TET3-"})
)

donor_col = 'donor_id'

# ----------------------------------------------------------------------
# 4. PREPARE COVARIATES 
# ----------------------------------------------------------------------
def filter_low_count_types(data, level_col, min_cells=50, min_donors=3):
    type_counts = data[level_col].value_counts()
    keep_cells = type_counts[type_counts >= min_cells].index
    
    keep = []
    for ct in keep_cells:
        donors_with = data[data[level_col] == ct][donor_col].nunique()
        if donors_with >= min_donors:
            keep.append(ct)
    return keep

def prepare_covariates(donor_metadata):
    covariates = pd.DataFrame()
    covariates['donor_id'] = donor_metadata['donor_id']
    
    # 1. Sex
    sex_map = {'PATO:0000384': 1, 'PATO:0000383': 0} 
    # Fallback checking ontology term or raw sex name
    if 'sex_ontology_term_id' in donor_metadata.columns:
        sex_col = donor_metadata['sex_ontology_term_id']
    else:
        sex_col = donor_metadata.get('sex', pd.Series())
        sex_map = {'male': 1, 'female': 0}
        
    sex_numeric = sex_col.astype(str).map(sex_map).fillna(0.5)
    covariates['Sex'] = pd.to_numeric(sex_numeric)
    
    # 2. Age
    age_mapping = {
        'Less than 65 years old': 55,
        '65 to 77 years old': 71,
        '78 to 89 years old': 83.5,
        '90+ years old': 93
    }
    age_numeric = donor_metadata['Age at death'].map(age_mapping)
    covariates['Age_numeric'] = pd.to_numeric(age_numeric, errors='coerce')
    covariates['Age_numeric'] = covariates['Age_numeric'].fillna(covariates['Age_numeric'].median())
    
    age_min = covariates['Age_numeric'].min()
    age_max = covariates['Age_numeric'].max()
    covariates['Age_normalized'] = (covariates['Age_numeric'] - age_min) / (age_max - age_min) if age_max > age_min else 0.5
    
    # 3. Race
    if 'self_reported_ethnicity_ontology_term_id' in donor_metadata.columns:
        race_col = donor_metadata['self_reported_ethnicity_ontology_term_id'].astype(str)
        covariates['Race_White'] = (race_col == 'HANCESTRO:0013').fillna(False).astype(int)
    else:
        race_col = donor_metadata.get('self_reported_ethnicity', pd.Series())
        covariates['Race_White'] = (race_col == 'European').fillna(False).astype(int)
        
    # 4. Genes Detected
    if 'Genes detected' in donor_metadata.columns:
        gd = donor_metadata['Genes detected'].astype(float)
        covariates['Genes_detected_scaled'] = (gd - gd.mean()) / gd.std()
    else:
        covariates['Genes_detected_scaled'] = 0.0
        
    # 5. Assay (Technology Batch)
    if 'assay' in donor_metadata.columns:
        covariates['assay'] = donor_metadata['assay'].astype(str)
        
    return covariates

def build_composition_data(data, level_col, min_cells=50, min_donors=3):
    keep = filter_low_count_types(data, level_col, min_cells, min_donors)
    if len(keep) < 2:
        return None, None
    data_f = data[data[level_col].isin(keep)]
    
    counts = data_f.groupby([donor_col, level_col]).size().unstack(fill_value=0)
    donor_metadata = data_f.groupby(donor_col).first().reset_index()
    covariates = prepare_covariates(donor_metadata)
    
    common_donors = counts.index.intersection(covariates['donor_id'])
    counts = counts.loc[common_donors]
    covariates = covariates[covariates['donor_id'].isin(common_donors)].set_index('donor_id')
    
    valid = [c for c in counts.columns if (counts[c] > 0).sum() >= min_donors]
    if len(valid) < 2:
        return None, None
    
    counts = counts[valid]
    return counts, covariates

# ----------------------------------------------------------------------
# 5. REFERENCE CYCLING (scCODA MODEL)
# ----------------------------------------------------------------------
def run_sccoda_with_reference(counts_df, covariates_df, reference_cell_type, 
                              clinical_var, ref_category, test_category,
                              fdr_level=0.05, num_results=20000, num_burnin=5000):
    try:
        counts_reset = counts_df.reset_index()
        covariates_reset = covariates_df.reset_index()
        merged = counts_reset.merge(covariates_reset, on='donor_id', how='inner')
        
        mask = (merged[clinical_var] == ref_category) | (merged[clinical_var] == test_category)
        merged = merged[mask].copy()
        
        if len(merged) < 3:
            return None
        
        # Binary target for scCODA
        merged['clinical_binary'] = (merged[clinical_var] == test_category).astype(int)
        merged = merged.set_index('donor_id')
        
        # Build Covariate Columns
        cov_cols = ['Sex', 'Age_normalized', 'Race_White', 'Genes_detected_scaled', 'clinical_binary']
        if 'assay' in merged.columns:
            cov_cols.append('assay')
            
        cell_cols = [c for c in counts_df.columns if c in merged.columns]
        sccoda_df = merged[cell_cols + cov_cols].copy()
        
        data_sccoda = dat.from_pandas(sccoda_df, covariate_columns=cov_cols)
        
        # Dynamic Formula Builder (Adds assay only if there are multiple tech batches in this comparison)
        fixed_effects = ['Sex', 'Age_normalized', 'Race_White', 'Genes_detected_scaled']
        if 'assay' in sccoda_df.columns and sccoda_df['assay'].nunique() > 1:
            fixed_effects.append('C(assay)')
            
        formula = " + ".join(fixed_effects) + " + clinical_binary"
        
        # Run Model
        model = mod.CompositionalAnalysis(
            data=data_sccoda,
            formula=formula,
            reference_cell_type=reference_cell_type
        )
        results = model.sample_hmc(num_results=num_results, num_burnin=num_burnin)
        results.set_fdr(est_fdr=fdr_level)
        
        credible = results.credible_effects()
        var_mask = credible.index.get_level_values('Covariate') == 'clinical_binary'
        credible_var = credible[var_mask]
        
        if len(credible_var) == 0:
            return None
        
        effects = results.effect_df.copy()
        var_effects = effects[var_mask].copy()
        var_effects.reset_index(inplace=True)
        var_effects.rename(columns={
            var_effects.columns[0]: 'covariate',
            var_effects.columns[1]: 'cell_type'
        }, inplace=True)
        
        var_effects['credible'] = credible_var.values
        
        sd_col = next((col for col in var_effects.columns if 'sd' in col.lower()), None)
        if sd_col is None:
            var_effects['sd'] = 0.0
        else:
            var_effects.rename(columns={sd_col: 'sd'}, inplace=True)
        
        var_effects['reference'] = reference_cell_type
        var_effects['test_category'] = test_category
        var_effects['ref_category'] = ref_category
        
        required_cols = ['cell_type', 'covariate', 'log2-fold change', 'sd', 'credible', 
                        'reference', 'test_category', 'ref_category']
        return var_effects[required_cols].copy()
    
    except Exception as e:
        print(f"      Model failed for reference '{reference_cell_type}': {e}")
        return None

def reference_cycling_for_comparison(counts_df, covariates_df, clinical_var, 
                                     ref_category, test_category, 
                                     fdr_level=0.05, mcmc_iter=20000):
    all_cell_types = sorted(counts_df.columns.tolist())
    if len(all_cell_types) < 2:
        return None
    
    print(f"    Cycling over {len(all_cell_types)} reference cell types...")
    all_results = []
    
    for i, ref in enumerate(all_cell_types, 1):
        print(f"      Reference {i}/{len(all_cell_types)}: {ref}")
        res_df = run_sccoda_with_reference(
            counts_df, covariates_df, ref, 
            clinical_var, ref_category, test_category,
            fdr_level, mcmc_iter
        )
        if res_df is not None:
            all_results.append(res_df)
    
    if not all_results:
        return None
    
    combined = pd.concat(all_results, ignore_index=True)
    summary = combined.groupby(['cell_type']).agg(
        log2fc_mean=('log2-fold change', 'mean'),
        log2fc_std=('log2-fold change', 'std'),
        n_references=('log2-fold change', 'count'),
        credible_prop=('credible', 'mean')
    ).reset_index()
    
    summary['credible_majority'] = summary['credible_prop'] >= 0.5
    summary['significant'] = summary['credible_majority']
    
    summary['clinical_var'] = clinical_var
    summary['ref_category'] = ref_category
    summary['test_category'] = test_category
    summary['fdr_threshold'] = fdr_level
    summary['comparison'] = f"{test_category} vs {ref_category}"
    
    return summary

def split_cell_type(cell_type):
    if '_TET3+' in cell_type: return cell_type.replace('_TET3+', ''), 'TET3+'
    if '_TET3-' in cell_type: return cell_type.replace('_TET3-', ''), 'TET3-'
    return cell_type, None

# ----------------------------------------------------------------------
# 6. PLOTTING
# ----------------------------------------------------------------------
def plot_forest_log2fc(summary_df, out_file, id_to_name=None):
    if summary_df.empty: return
    
    sub = summary_df[summary_df['significant']].copy()
    if sub.empty: return
    
    sub = sub.dropna(subset=['log2fc_mean', 'log2fc_std'])
    sub[['base_id', 'status']] = sub['cell_type'].apply(lambda x: pd.Series(split_cell_type(x)))
    sub = sub.dropna(subset=['status'])
    
    if id_to_name: sub['base_name'] = sub['base_id'].map(id_to_name).fillna(sub['base_id'])
    else: sub['base_name'] = sub['base_id']
    
    available_bases = [b for b in CUSTOM_CELL_ORDER if b in sub['base_name'].values]
    missing_bases = [b for b in sub['base_name'].unique() if b not in available_bases]
    final_order = available_bases + sorted(missing_bases)
    
    sub['base_name'] = pd.Categorical(sub['base_name'], categories=final_order, ordered=True)
    sub = sub.sort_values('base_name')
    y_positions = {base: i for i, base in enumerate(reversed(final_order))}
    
    plt.figure(figsize=(10, max(4, len(final_order)*0.4)))
    color_map = {'TET3+': 'red', 'TET3-': 'blue'}
    
    for _, row in sub.iterrows():
        y = y_positions[row['base_name']]
        x = row['log2fc_mean']
        color = color_map.get(row['status'], 'gray')
        plt.errorbar(x, y, xerr=row['log2fc_std'], fmt='none', ecolor='black', capsize=3, alpha=0.7)
        plt.plot(x, y, 'o', color=color, alpha=0.8, markersize=8, markeredgecolor='black')
    
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    plt.xlabel('Mean log₂‑fold change (±1 SD)')
    plt.title(f"A9 – {sub['comparison'].iloc[0]}\n(Averaged across reference cycles)")
    
    plt.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    test_cat = sub['test_category'].iloc[0]
    ax.text(0.02, -0.12, f'lower in {test_cat}', transform=ax.transAxes, ha='left', va='top', fontsize=9, color='gray')
    ax.text(0.98, -0.12, f'higher in {test_cat}', transform=ax.transAxes, ha='right', va='top', fontsize=9, color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved forest plot: {out_file}")

# ----------------------------------------------------------------------
# 7. MAIN EXECUTOR
# ----------------------------------------------------------------------
def main():
    np.random.seed(12345)
    
    print("\n" + "="*70)
    print("BUILDING COMPOSITION DATA - A9 REGION")
    print("="*70)
    
    counts_df, covariates_df = build_composition_data(
        obs_df,
        level_col='cell_type_subclass_tet3',
        min_cells=MIN_CELLS_PER_TYPE,
        min_donors=MIN_DONORS_PER_TYPE
    )
    
    if counts_df is None or covariates_df is None:
        print("❌ Could not build composition data. Exiting.")
        return
    
    print(f"\n✅ Composition data built successfully:")
    print(f"   Donors: {len(counts_df)}")
    print(f"   Covariates generated: {list(covariates_df.columns)}")
    
    donor_info = pd.DataFrame({
        'donor_id': counts_df.index,
        'n_cells_TET3+': (counts_df.filter(like='TET3+').sum(axis=1)),
        'n_cells_TET3-': (counts_df.filter(like='TET3-').sum(axis=1))
    })
    donor_info.to_csv(RESULTS_DIR / "donor_summary.csv", index=False)
    
    if 'ADNC' in obs_df.columns:
        covariates_df['ADNC'] = covariates_df.index.map(obs_df.groupby('donor_id')['ADNC'].first())
    
    all_summaries = []
    
    # --- ADNC ANALYSIS ---
    if 'ADNC' in covariates_df.columns:
        print("\n" + "="*70 + "\nADNC ANALYSIS\n" + "="*70)
        for test_category in ADNC_TEST_CATEGORIES:
            valid_mask = (covariates_df['ADNC'] == ADNC_REFERENCE) | (covariates_df['ADNC'] == test_category)
            summary = reference_cycling_for_comparison(
                counts_df[valid_mask], covariates_df[valid_mask].copy(), 'ADNC',
                ADNC_REFERENCE, test_category, fdr_level=FDR_LEVEL, mcmc_iter=MCMC_ITER
            )
            
            if summary is not None and not summary.empty:
                summary.to_csv(RESULTS_DIR / f"summary_ADNC_{test_category}_vs_{ADNC_REFERENCE}.csv", index=False)
                all_summaries.append(summary)
                plot_forest_log2fc(summary, RESULTS_DIR / f"forest_ADNC_{test_category}_vs_{ADNC_REFERENCE}.png", CELL_TYPE_RENAME)
    
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(RESULTS_DIR / "all_comparisons_summary.csv", index=False)
        print(f"\n✅ Combined summary saved: {RESULTS_DIR / 'all_comparisons_summary.csv'}")

if __name__ == "__main__":
    main()