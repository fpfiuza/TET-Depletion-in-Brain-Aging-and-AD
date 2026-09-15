"""
Script to extract the percentages of TET-expressing cells in each cell type per donor.
It reads a h5ad file (MTG or A9 brain regions) available at the Seattle Alzheimer's Disease Brain Cell Atlas(https://brain-map.org/consortia/sea-ad) - dataset of Gabitto et al. 2024 (https://doi.org/10.1038/s41593-024-01774-5).  
     
"""

import numpy as np
import pandas as pd
import h5py
import os

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
H5AD_PATH = "dad4819b-4c14-439c-b32a-2c8d68bd22e1.h5ad"  
OUTPUT_EXPR_CSV = "A9_TET_expression_summary.csv"

# Metadata Column Names inside the h5ad file (.obs dataframe)
H5AD_DONOR_COL = "donor_id"
H5AD_SUBCLASS_COL = "Subclass"

def read_obs_field(f, field_name):
    """Safely reads a column from the .obs group, handling categorical and group encodings."""
    import h5py
    
    obs_group = f['obs']
    if field_name not in obs_group:
        raise KeyError(f"Field '{field_name}' not found in /obs")
    
    ds = obs_group[field_name]
    
    if isinstance(ds, h5py.Group):
        if 'codes' in ds and 'categories' in ds:
            codes = ds['codes'][...]
            cats = ds['categories'][...]
            cats = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cats]
            return [cats[code] if code >= 0 else None for code in codes]
        elif 'codes' in ds:
            codes = ds['codes'][...]
            if 'obs/__categories' in f and field_name in f['obs/__categories']:
                cats = f[f'obs/__categories/{field_name}'][...]
            elif f"uns/{field_name}_categories" in f:
                cats = f[f"uns/{field_name}_categories"][...]
            else:
                raise ValueError(f"Categorical field '{field_name}' group contains 'codes' but no categories found.")
            cats = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cats]
            return [cats[code] if code >= 0 else None for code in codes]
        else:
            raise TypeError(f"Field '{field_name}' is an HDF5 Group but does not follow recognized categorical format. Keys: {list(ds.keys())}")
            
    if 'obs/__categories' in f and field_name in f['obs/__categories']:
        cats = f[f'obs/__categories/{field_name}'][...]
        cats = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cats]
        codes = ds[...]
        return [cats[code] if code >= 0 else None for code in codes]
    
    if f"uns/{field_name}_categories" in f:
        cats = f[f"uns/{field_name}_categories"][...]
        cats = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cats]
        codes = ds[...]
        return [cats[code] if code >= 0 else None for code in codes]
        
    data = ds[...]
    if hasattr(data, 'dtype') and data.dtype.kind in ['O', 'S', 'U']: 
        return [d.decode('utf-8') if isinstance(d, bytes) else str(d) for d in data]
    return data

def main():
    print(f"--- Step 1: Streaming .h5ad structure from disk using h5py ---")
    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(f"Could not find .h5ad file at: '{H5AD_PATH}'")
    
    with h5py.File(H5AD_PATH, 'r') as f:
        if '_index' in f['obs'].attrs:
            idx_col = f['obs'].attrs['_index']
            idx_col = idx_col.decode('utf-8') if isinstance(idx_col, bytes) else idx_col
            cell_names_raw = f[f'obs/{idx_col}'][...]
        elif '_index' in f['obs']:
            cell_names_raw = f['obs/_index'][...]
        else:
            cell_names_raw = f['obs/index'][...]
        
        cell_names = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in cell_names_raw]
        num_cells = len(cell_names)
        print(f"File contains metadata for {num_cells} individual cells.")
        
        print("Extracting cell classifications and donor assignments...")
        donor_ids = read_obs_field(f, H5AD_DONOR_COL)
        subclasses = read_obs_field(f, H5AD_SUBCLASS_COL)
        
        var_path = 'raw/var' if 'raw/var' in f else 'var'
        var_group = f[var_path]
        if '_index' in var_group.attrs:
            g_idx_col = var_group.attrs['_index']
            g_idx_col = g_idx_col.decode('utf-8') if isinstance(g_idx_col, bytes) else g_idx_col
            gene_names_raw = var_group[g_idx_col][...]
        elif '_index' in var_group:
            gene_names_raw = var_group['_index'][...]
        else:
            gene_names_raw = var_group['index'][...]
            
        gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in gene_names_raw]
        
        genes = ['TET1', 'TET2', 'TET3']
        gene_to_idx = {}
        
        ensembl_fallback = {
            'TET1': 'ENSG00000138336',
            'TET2': 'ENSG00000168769',
            'TET3': 'ENSG00000187605'
        }
        
        for g in genes:
            if g in gene_names:
                gene_to_idx[g] = gene_names.index(g)
                
        missing_genes = [g for g in genes if g not in gene_to_idx]
        if missing_genes:
            for key in var_group.keys():
                if key.startswith('__'): continue
                if isinstance(var_group[key], h5py.Dataset):
                    try:
                        col_data = var_group[key][...]
                        col_strings = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in col_data]
                        for g in list(missing_genes):
                            if g in col_strings:
                                gene_to_idx[g] = col_strings.index(g)
                                missing_genes.remove(g)
                    except Exception:
                        pass
                        
        missing_genes = [g for g in genes if g not in gene_to_idx]
        if missing_genes:
            for g in list(missing_genes):
                target_ens = ensembl_fallback[g]
                for idx, name in enumerate(gene_names):
                    clean_name = name.split('.')[0]
                    if clean_name == target_ens:
                        gene_to_idx[g] = idx
                        missing_genes.remove(g)
                        break
                        
        for g in genes:
            if g not in gene_to_idx:
                raise KeyError(f"Target gene '{g}' could not be resolved.")

        x_path = 'raw/X' if 'raw/X' in f else 'X'
        x_node = f[x_path]
        
        gene_exprs = {g: np.zeros(num_cells, dtype=np.float32) for g in genes}
        
        if isinstance(x_node, h5py.Dataset):
            for g, idx in gene_to_idx.items():
                gene_exprs[g] = x_node[:, idx]
        else:
            encoding = x_node.attrs.get('encoding-type', b'').decode('utf-8') if isinstance(x_node.attrs.get('encoding-type'), bytes) else x_node.attrs.get('encoding-type', '')
            if not encoding and 'indptr' in x_node:
                encoding = 'csr_matrix'
                
            indptr = x_node['indptr'][...]
            if 'csc_matrix' in encoding:
                for g, idx in gene_to_idx.items():
                    p_start, p_end = indptr[idx], indptr[idx+1]
                    gene_exprs[g][x_node['indices'][p_start:p_end]] = x_node['data'][p_start:p_end]
            else:
                chunk_size = 100000
                for start_idx in range(0, num_cells, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_cells)
                    p_start, p_end = indptr[start_idx], indptr[end_idx]
                    
                    chunk_indices = x_node['indices'][p_start:p_end]
                    chunk_data = x_node['data'][p_start:p_end]
                    
                    row_counts = indptr[start_idx+1:end_idx+1] - indptr[start_idx:end_idx]
                    chunk_rows = np.repeat(np.arange(start_idx, end_idx), row_counts)
                    
                    for g, idx in gene_to_idx.items():
                        mask = (chunk_indices == idx)
                        if np.any(mask):
                            gene_exprs[g][chunk_rows[mask]] = chunk_data[mask]

        print("--- Step 2: Formulating cellular expression dataframes ---")
        df_cells = pd.DataFrame(gene_exprs, index=cell_names)
        df_cells['donor_id'] = donor_ids
        df_cells['subclass'] = subclasses

    print("Aggregating single-cell counts to donor-level metrics per subclass...")
    agg_list = []
    grouped = df_cells.groupby(['subclass', 'donor_id'])
    
    for (subclass, donor_id), sub_df in grouped:
        if pd.isna(subclass) or pd.isna(donor_id):
            continue
        row = {'subclass': subclass, 'donor_id': donor_id}
        for gene in genes:
            row[f'{gene}_pct_expressing'] = (sub_df[gene] > 0).mean() * 100
        agg_list.append(row)
        
    df_expr_all = pd.DataFrame(agg_list)
    df_expr_all.to_csv(OUTPUT_EXPR_CSV, index=False)
    print(f"\n[SUCCESS] Extraction complete. Results exported to: '{OUTPUT_EXPR_CSV}'")

if __name__ == "__main__":
    main()