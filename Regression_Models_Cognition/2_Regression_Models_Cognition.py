"""
Multivariate regression models to evaluate associations of cell percentages with cognitive scores. Based
on the Seattle Alzheimer's Disease Brain Cell Atlas(https://brain-map.org/consortia/sea-ad) - dataset of Gabitto et al. 2024 (https://doi.org/10.1038/s41593-024-01774-5) and Mukherjee et al. 2020 (https://doi.org/10.1038/s41380-018-0298-8).

The user should first download the "Donor metadata" (which was renamed here as SEAD_donor_metadata) and the "Harmonized cognitive scores" (sea-ad_cohort_harmonized_cognitive_scores) xlsx files from the https://brain-map.org/consortia/sea-ad/our-data website. The user also should run the "Extract_TET_CellPercentages.py" script prior to this one.  

"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
EXPR_CSV_PATH = "A9_TET_expression_summary.csv"
GLOBAL_COGNITION_PATH = "SEAD_donor_metadata.xlsx"
DOMAIN_COGNITION_PATH = "sea-ad_cohort_harmonized_cognitive_scores.xlsx"
OUTPUT_CSV = "TET_cell_subclass_regression_results.csv"

EXCLUDE_DONORS = []
REMOVE_EXTREME_OUTLIERS = True
OUTLIER_Z_THRESHOLD = 3.0

# ==========================================
# MODULAR COVARIATE SELECTION
# ==========================================
MODULAR_COVARIATES = [
    'age_at_death',
    'C(sex)',
    'C(Cohort)'
]

INCLUDE_TEST_INTERVAL = True


def main():
    if not os.path.exists(EXPR_CSV_PATH):
        raise FileNotFoundError(
            f"Could not find expression CSV at: '{EXPR_CSV_PATH}'. Run the extraction script first."
        )

    print('--- Step 1: Loading aggregated expression data ---')
    df_expr_all = pd.read_csv(EXPR_CSV_PATH)

    print('--- Step 2: Merging clinical and cognitive metadata sheets ---')
    if GLOBAL_COGNITION_PATH.endswith('.csv'):
        df_global = pd.read_csv(GLOBAL_COGNITION_PATH)
    else:
        df_global = pd.read_excel(GLOBAL_COGNITION_PATH)

    df_global_clean = df_global.rename(
        columns={
            'Donor ID': 'donor_id',
            'Cognitive Status': 'cognitive_status',
            'Last MMSE Score': 'MMSE',
            'Age at Death': 'age_at_death',
            'Age at Death (years)': 'age_at_death',
            'Sex': 'sex',
            'Primary Study Name': 'Cohort',
            'Interval from last MMSE in months': 'interval_MMSE',
            'Interval from last MMSE (months)': 'interval_MMSE',
        }
    )

    df_global_clean['cognitive_status'] = (
        df_global_clean['cognitive_status']
        .astype(str)
        .str.strip()
        .replace({
            'No dementia': 'Normal Cognition',
            'Dementia': "Alzheimer's Disease"
        })
    )

    df_domain = pd.read_excel(DOMAIN_COGNITION_PATH)
    df_domain_clean = df_domain.rename(columns={'Donor ID': 'donor_id'})
    df_domain_clean = df_domain_clean.sort_values(by=['donor_id', 'age_vis'])
    df_domain_last = df_domain_clean.drop_duplicates(
        subset=['donor_id'], keep='last'
    ).copy()

    clinical_meta = pd.merge(
        df_global_clean, df_domain_last, on='donor_id', how='inner'
    )
    clinical_meta['interval_domains'] = (
        np.abs(clinical_meta['age_at_death'] - clinical_meta['age_vis']) * 12
    )

    models_config = {
        'MMSE': ('interval_MMSE', 'Global MMSE Score'),
        'MEM_E': ('interval_domains', 'Memory Domain (MEM_E)'),
        'EXF_E': ('interval_domains', 'Executive Function (EXF_E)'),
        'LAN_E': ('interval_domains', 'Language Domain (LAN_E)'),
        'VSP_E': ('interval_domains', 'Visuospatial Domain (VSP_E)'),
    }

    metrics_config = {'pct_expressing': 'Cell Percentage (%)'}

    genes = ['TET1', 'TET2', 'TET3']
    results_list = []
    subclasses_list = sorted(df_expr_all['subclass'].unique())

    print('\n=====================================================================')
    print('                 STATISTICAL MODEL SUMMARY                           ')
    print('=====================================================================')
    print('Model Type: Ordinary Least Squares (OLS) Multiple Regression')
    print('Primary Equation:')
    print('  Score ~ Predictor * Cognitive_Status + COVARIATES\n')
    print('Variables Defined:')
    print(
        '  • Dependent Variable (Score): Configured Cognitive Scores'
        f' {list(models_config.keys())}'
    )
    print('  • Predictor: Gene expression (% Expressing of TET1, 2, 3)')
    print(
        '  • Interaction Term: Predictor : Cognitive Status (Reference: Normal Cognition)'
    )
    print(f'  • Fixed Covariates: {MODULAR_COVARIATES}')
    print('  • Optional Covariate: Test Interval (if applicable to the specific score)\n')
    print('Interpretation of Coefficients in Output:')
    print("  1. 'Predictor' -> Beta slope for the 'Normal Cognition' baseline group.")
    print(
        "  2. 'Predictor:C(cognitive_status)[T.Alzheimer\\'s Disease]' -> The difference in slope between AD and Normal groups (Interaction effect)."
    )
    print("  3. 'Calculated AD Slope' -> Predictor slope + Interaction effect (Direct slope for the AD group).")
    print('=====================================================================\n')

    print('--- Step 3: Computing multi-variable interactions across all subclasses ---')
    for subclass in subclasses_list:
        df_subclass_expr = df_expr_all[df_expr_all['subclass'] == subclass]
        merged = pd.merge(
            df_subclass_expr, clinical_meta, on='donor_id', how='inner'
        )

        if len(EXCLUDE_DONORS) > 0:
            merged = merged[~merged['donor_id'].isin(EXCLUDE_DONORS)].copy()

        cohort_df = merged[
            merged['cognitive_status'].isin(['Normal Cognition', "Alzheimer's Disease"])
        ].copy()

        if cohort_df.empty:
            continue

        for gene in genes:
            for metric_suffix, metric_label in metrics_config.items():
                iv_col = f'{gene}_{metric_suffix}'

                for score, (interval_col, score_label) in models_config.items():
                    covariates = []
                    if INCLUDE_TEST_INTERVAL and interval_col in cohort_df.columns:
                        covariates.append(interval_col)

                    for cov in MODULAR_COVARIATES:
                        raw_cov = cov.replace('C(', '').replace(')', '')
                        if raw_cov not in cohort_df.columns:
                            continue
                        if cohort_df[raw_cov].nunique() <= 1:
                            continue
                        if cov not in covariates and raw_cov != interval_col:
                            covariates.append(cov)

                    formula = (
                        f'{score} ~ {iv_col} * C(cognitive_status, Treatment(reference=\'Normal Cognition\'))'
                        + (f' + ' + ' + '.join(covariates) if covariates else '')
                    )

                    try:
                        raw_covariates = [
                            c.replace('C(', '').replace(')', '') for c in covariates
                        ]
                        cols_needed = [score, iv_col, 'cognitive_status'] + raw_covariates
                        df_clean = cohort_df.dropna(subset=cols_needed).copy()

                        if REMOVE_EXTREME_OUTLIERS and not df_clean.empty:
                            df_clean = df_clean[df_clean[iv_col] > 0]
                            m_val = df_clean[iv_col].mean()
                            s_val = df_clean[iv_col].std()
                            if s_val > 0:
                                z_scores = np.abs((df_clean[iv_col] - m_val) / s_val)
                                df_clean = df_clean[z_scores <= OUTLIER_Z_THRESHOLD]

                        if len(df_clean) < 5:
                            continue

                        model = smf.ols(formula, data=df_clean).fit()

                        for param_name in model.params.index:
                            results_list.append({
                                'Cell_Subclass': subclass,
                                'Cohort': 'Interaction Model',
                                'Enzyme': gene,
                                'Predictor_Metric': metric_label,
                                'Cognitive_Score': score,
                                'Covariate': param_name,
                                'Coefficient': model.params[param_name],
                                'Standard_Error': model.bse[param_name],
                                'Raw_P_Value': model.pvalues[param_name],
                                'CI_Lower': model.conf_int().loc[param_name][0],
                                'CI_Upper': model.conf_int().loc[param_name][1],
                                'Model_Observations': int(model.nobs),
                                'Model_R_Squared': model.rsquared,
                                'Model_Adj_R_Squared': model.rsquared_adj,
                            })

                        interaction_term = (
                            f'{iv_col}:C(cognitive_status, Treatment(reference=\'Normal Cognition\'))[T.Alzheimer\'s Disease]'
                        )
                        if interaction_term in model.params:
                            ad_slope_test = model.t_test(f'{iv_col} + {interaction_term} = 0')
                            results_list.append({
                                'Cell_Subclass': subclass,
                                'Cohort': 'Interaction Model',
                                'Enzyme': gene,
                                'Predictor_Metric': metric_label,
                                'Cognitive_Score': score,
                                'Covariate': f'{iv_col} (Calculated AD Slope)',
                                'Coefficient': ad_slope_test.effect.item(),
                                'Standard_Error': ad_slope_test.sd.item(),
                                'Raw_P_Value': ad_slope_test.pvalue.item(),
                                'CI_Lower': ad_slope_test.conf_int()[0, 0],
                                'CI_Upper': ad_slope_test.conf_int()[0, 1],
                                'Model_Observations': int(model.nobs),
                                'Model_R_Squared': model.rsquared,
                                'Model_Adj_R_Squared': model.rsquared_adj,
                            })

                    except Exception as e:
                        pass

    print('--- Step 4: Applying FDR Correction ---')
    if results_list:
        df_results = pd.DataFrame(results_list)
        df_results['Adjusted_P_Value_FDR'] = np.nan

        def categorize_term(cov_name):
            if 'Calculated AD Slope' in cov_name:
                return 'Calculated_AD_Slope'
            elif ':' in cov_name:
                return 'Interaction_Term'
            elif any(g in cov_name for g in ['TET1', 'TET2', 'TET3']):
                return 'Baseline_Gene_Slope'
            else:
                return 'Nuisance_Covariate'

        df_results['Term_Type'] = df_results['Covariate'].apply(categorize_term)
        primary_mask = df_results['Term_Type'] != 'Nuisance_Covariate'
        fdr_grouping = ['Cognitive_Score', 'Term_Type']

        for names, group_df in df_results[primary_mask].groupby(fdr_grouping):
            raw_pvals = group_df['Raw_P_Value'].values
            if len(raw_pvals) > 0:
                _, pvals_adj, _, _ = multipletests(
                    raw_pvals, alpha=0.05, method='fdr_bh'
                )
                df_results.loc[group_df.index, 'Adjusted_P_Value_FDR'] = pvals_adj

        cols = list(df_results.columns)
        cols.remove('Adjusted_P_Value_FDR')
        raw_p_idx = cols.index('Raw_P_Value')
        cols.insert(raw_p_idx + 1, 'Adjusted_P_Value_FDR')
        df_results = df_results[cols]

        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[SUCCESS] Analysis complete. Results exported to: '{OUTPUT_CSV}'")
    else:
        print('Warning: No regression models could be fit successfully for any subclass.')


if __name__ == '__main__':
    main()