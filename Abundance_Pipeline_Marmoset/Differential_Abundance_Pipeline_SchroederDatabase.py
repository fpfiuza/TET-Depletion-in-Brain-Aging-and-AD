"""
Comprehensive scCODA analysis of cell proportions in marmoset, 
based on the dataset of Schroeder et al. 2025 (https://doi.org/10.1016/j.neuron.2025.09.011).
Evaluates the TET-split(or any gene-split) abundance for a single brain region. In this case, TET3 is being evaluated.
The user should first download the .h5ad file available at https://singlecell.broadinstitute.org/single_cell/study/SCP2706/a-multi-region-transcriptomic-atlas-of-developmental-cell-type-diversity-in-marmoset-brain

This script runs the unified dataset systematically shifting the baseline 
reference to extract every possible pairwise age comparison.
It utilizes Reference cycling (majority voting) for each baseline model.

MCMC parameters: 20000 iterations; 5000 burn-in
"""

import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# scCODA imports
# ----------------------------------------------------------------------
from sccoda.util import comp_ana as mod
from sccoda.util import cell_composition_data as dat

# ----------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------
DATA_FILE = "adata_marm_allages_allregions_forBroadSCPortal.h5ad"

# ---> CHANGE THESE TWO LINES FOR OTHER REGIONS <---
RESULTS_DIR = Path("results/scCODA_TET3_refcycled_PFC")
REGIONS = ["PFC"] 
# --------------------------------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
TET3_THRESHOLD = 0.5
MIN_CELLS_PER_TYPE = 50
MIN_DONORS_PER_TYPE = 3
MCMC_ITER = 20000
MCMC_BURNIN = 5000
FDR_LEVEL = 0.05

AGE_ORDER = ["GD135", "neonate", "7months", "14months", "30months", "aged"]

# Restricted to ONLY tet3-split analysis
ANALYSIS_MODES = [
    ('tet3_split', 'cell_type_broad_tet3')
]

# ----------------------------------------------------------------------
# 1. LOAD DATA & EXTRACT TET3
# ----------------------------------------------------------------------
print(f"Loading AnnData for {REGIONS[0]} analysis...")
adata = sc.read_h5ad(DATA_FILE)

possible_names = ["TET3", "Tet3", "tet3"]
tet3_gene = next((name for name in possible_names if name in adata.var_names), None)
if tet3_gene is None: raise ValueError("TET3 gene not found in var_names.")

tet3_expr = adata[:, tet3_gene].X
if hasattr(tet3_expr, "toarray"): tet3_expr = tet3_expr.toarray().flatten()
else: tet3_expr = tet3_expr.flatten()
adata.obs["TET3_positive"] = (tet3_expr > TET3_THRESHOLD).astype(bool)

# ----------------------------------------------------------------------
# 2. DEFINE CELL TYPES
# ----------------------------------------------------------------------
def map_leiden_to_broad(leiden_name):
    if pd.isna(leiden_name): return "Unknown"
    if leiden_name.startswith("Neuron_Excit"): return "NeuronExcit"
    if leiden_name.startswith("Neuron_Inh"): return "NeuronInh"
    if leiden_name.startswith("Neuron_Mixed"): return "NeuronMixed"
    if "Astrocyte" in leiden_name: return "Astrocyte"
    if "Oligodendrocyte" in leiden_name: return "Oligodendrocyte"
    if "OPC" in leiden_name: return "OPC"
    if "Microglia" in leiden_name: return "Microglia"
    if "Endothelial" in leiden_name: return "Endothelial"
    if "Ependymal" in leiden_name: return "Ependymal"
    if "Fibroblast" in leiden_name: return "Fibroblast"
    if "Mural" in leiden_name: return "Mural"
    if "Choroid" in leiden_name or "ChoroidPlexus" in leiden_name: return "ChoroidPlexus"
    return "Other"

adata.obs["cell_type_broad"] = adata.obs["leiden"].apply(map_leiden_to_broad)

id_to_name = {
    "NeuronExcit": "Excitatory neuron", "NeuronInh": "Inhibitory neuron",
    "NeuronMixed": "Mixed neuron", "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligodendrocyte", "OPC": "OPC",
    "Microglia": "Microglia", "Endothelial": "Endothelial",
    "Ependymal": "Ependymal", "Fibroblast": "Fibroblast",
    "Mural": "Mural", "ChoroidPlexus": "Choroid plexus"
}

adata.obs["cell_type_broad_tet3"] = (
    adata.obs["cell_type_broad"].astype(str) + "_" + 
    adata.obs["TET3_positive"].map({True: "TET3+", False: "TET3-"})
)

# ----------------------------------------------------------------------
# 3. DONOR & REGION COLUMNS
# ----------------------------------------------------------------------
donor_col = next((col for col in ['donor_id', 'replicate'] if col in adata.obs.columns), None)
if donor_col is None: raise KeyError("No donor column found")

if 'region_dissected' in adata.obs.columns: adata.obs['region_analysis'] = adata.obs['region_dissected']
elif 'regions_broad' in adata.obs.columns: adata.obs['region_analysis'] = adata.obs['regions_broad']
elif 'region' in adata.obs.columns: adata.obs['region_analysis'] = adata.obs['region']
else: raise KeyError("No region column found.")

# ----------------------------------------------------------------------
# 4. HELPER FUNCTIONS FOR COMPOSITION AND MODELING
# ----------------------------------------------------------------------
def filter_low_count_types(data, level_col, min_cells, min_donors):
    type_counts = data[level_col].value_counts()
    keep_cells = type_counts[type_counts >= min_cells].index
    keep = [ct for ct in keep_cells if data[data[level_col] == ct][donor_col].nunique() >= min_donors]
    return keep

def build_composition_data(data, level_col, min_cells, min_donors):
    keep = filter_low_count_types(data, level_col, min_cells, min_donors)
    if len(keep) < 2: return None
    data_f = data[data[level_col].isin(keep)]

    counts = data_f.groupby([donor_col, level_col]).size().unstack(fill_value=0)
    metadata = data_f.groupby(donor_col).agg({"age": "first", "sex": "first"})
    
    mean_genes = data_f.groupby(donor_col)['n_genes'].mean()
    metadata['n_genes_scaled'] = (mean_genes - mean_genes.mean()) / mean_genes.std()
    metadata['n_genes_scaled'] = metadata['n_genes_scaled'].fillna(0.0)

    combined = counts.join(metadata, how='inner').dropna(subset=["age", "sex"])
    valid = [c for c in counts.columns if (combined[c] > 0).sum() >= min_donors]
    
    if len(valid) < 2: return None
    return combined[valid + ["age", "sex", "n_genes_scaled"]]

def run_sccoda_with_baseline_and_ref(df, baseline_age, reference_cell_type, valid_targets, fdr_level, num_results, num_burnin):
    try:
        cov_cols = ["age", "sex", "n_genes_scaled"]
        adata_coda = dat.from_pandas(df, covariate_columns=cov_cols)
        
        formula = f"C(age, Treatment('{baseline_age}')) + sex + n_genes_scaled"
        
        model = mod.CompositionalAnalysis(
            data=adata_coda,
            formula=formula,
            reference_cell_type=reference_cell_type
        )
        
        results = model.sample_hmc(num_results=num_results, num_burnin=num_burnin)
        results.set_fdr(est_fdr=fdr_level)
        
        effect_df = results.effect_df.copy()
        credible_df = results.credible_effects()
        
        age_mask = effect_df.index.get_level_values('Covariate').str.contains('age', case=False)
        age_effects = effect_df[age_mask].copy()
        age_credible = credible_df[age_mask].copy()
        
        if age_effects.empty: return None

        age_effects = age_effects.reset_index()
        age_effects['is_credible'] = age_credible.values 
        
        def clean_cov_name(name):
            if "[T." in name:
                return name.split("[T.")[-1].replace("]", "")
            return name
            
        age_effects['comparison_group'] = age_effects['Covariate'].apply(clean_cov_name)
        age_effects['baseline_group'] = baseline_age
        
        age_effects = age_effects[age_effects['comparison_group'].isin(valid_targets)].copy()
        return age_effects

    except Exception as e:
        print(f"      scCODA model failed for baseline {baseline_age} (ref: {reference_cell_type}): {e}")
        return None

def reference_cycling_baseline(df, baseline_age, valid_targets, fdr_level, mcmc_iter, mcmc_burnin):
    cell_types = sorted([c for c in df.columns if c not in ['age', 'sex', 'n_genes_scaled']])
    if len(cell_types) < 2: return None

    print(f"    Cycling over {len(cell_types)} reference cell types...")
    all_runs = []

    for ref in cell_types:
        res_df = run_sccoda_with_baseline_and_ref(
            df, baseline_age, ref, valid_targets, fdr_level, mcmc_iter, mcmc_burnin
        )
        if res_df is not None and not res_df.empty:
            res_df['reference_used'] = ref
            all_runs.append(res_df)

    if not all_runs: return None

    combined_df = pd.concat(all_runs, ignore_index=True)
    summary_rows = []

    for (comp_group, cell_type), group in combined_df.groupby(['comparison_group', 'Cell Type']):
        valid_group = group[group['reference_used'] != cell_type]
        if len(valid_group) == 0: continue
            
        times_cred = valid_group['is_credible'].sum()
        total_refs = len(cell_types) 
        
        pct_cred = times_cred / total_refs
        is_cred_final = pct_cred >= 0.5
        
        eff_mean = valid_group['Final Parameter'].mean()
        log2fc_mean = valid_group['log2-fold change'].mean()
        log2fc_std = valid_group['log2-fold change'].std()
        
        summary_rows.append({
            'Cell Type': cell_type,
            'baseline_group': baseline_age,
            'comparison_group': comp_group,
            'times_credible': times_cred,
            'pct_credible': pct_cred,
            'is_credible': is_cred_final,
            'Final Parameter': eff_mean,
            'log2-fold change': log2fc_mean,
            'log2fc_std': log2fc_std,
            'n_refs_used': len(valid_group)
        })

    return pd.DataFrame(summary_rows)

# ----------------------------------------------------------------------
# 5. MAIN LOOP
# ----------------------------------------------------------------------
def main():
    np.random.seed(12345)
    all_summaries = []

    for region in REGIONS:
        print("\n" + "="*70)
        print(f"REGION: {region}")
        print("="*70)

        region_mask = (adata.obs['region_analysis'] == region)
        obs_region = adata.obs[region_mask].copy()
        
        if len(obs_region) == 0:
            print("  No cells. Skipping.")
            continue

        for mode_name, level_col in ANALYSIS_MODES:
            print(f"\n  --- MODE: {mode_name} ---")
            
            comp_df = build_composition_data(obs_region, level_col, MIN_CELLS_PER_TYPE, MIN_DONORS_PER_TYPE)
            if comp_df is None:
                print("    Not enough data. Skipping.")
                continue
                
            available_ages = [age for age in AGE_ORDER if age in comp_df['age'].unique()]
            
            if len(available_ages) < 2:
                print("    Not enough age categories for comparison. Skipping.")
                continue

            for i, baseline_age in enumerate(available_ages[:-1]):
                valid_targets = available_ages[i+1:]
                print(f"    Running Baseline: {baseline_age} -> vs {valid_targets}")
                
                results_df = reference_cycling_baseline(
                    df=comp_df, baseline_age=baseline_age, valid_targets=valid_targets,
                    fdr_level=FDR_LEVEL, mcmc_iter=MCMC_ITER, mcmc_burnin=MCMC_BURNIN
                )

                if results_df is not None and not results_df.empty:
                    results_df['mode'] = mode_name
                    results_df['region'] = region
                    
                    results_df['cell_type_base'] = results_df['Cell Type'].apply(lambda x: x.split('_')[0])
                    results_df['cell_type_nice'] = results_df['cell_type_base'].map(id_to_name).fillna(results_df['cell_type_base'])
                    
                    def get_tet3_status(ct):
                        if '_TET3+' in ct: return 'TET3+'
                        elif '_TET3-' in ct: return 'TET3-'
                        return 'All'
                    results_df['tet3_status'] = results_df['Cell Type'].apply(get_tet3_status)
                    all_summaries.append(results_df)

    if all_summaries:
        master_df = pd.concat(all_summaries, ignore_index=True)
        
        cols = ['region', 'mode', 'baseline_group', 'comparison_group', 'Cell Type', 'cell_type_base', 
                'cell_type_nice', 'tet3_status', 'times_credible', 'pct_credible', 'is_credible', 
                'Final Parameter', 'log2-fold change', 'log2fc_std', 'n_refs_used']
        
        master_df = master_df[[c for c in cols if c in master_df.columns] + [c for c in master_df.columns if c not in cols]]
        
        # Save output with region specific name
        master_file = RESULTS_DIR / f"scCODA_results_{REGIONS[0]}_tet3_split.csv"
        master_df.to_csv(master_file, index=False)
        print(f"\n✅ Saved master results with {len(master_df)} rows to {master_file}")

    print("\n" + "="*70)
    print(f"ALL DONE FOR {REGIONS[0]}. Results saved in:", RESULTS_DIR)
    print("="*70)

if __name__ == "__main__":
    main()