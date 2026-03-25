
import nbformat as nbf
import numpy as np

def add_ensemble_strategies(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Add Task 3: Ensemble Strategies
    new_markdown = nbf.v4.new_markdown_cell(
        "## Task 3: Ensemble Strategies\n\n"
        "We now test two ensemble strategies:\n"
        "1. **Hourly + Cluster + Global (H+C+G)**: Combining all three modeling architectures\n"
        "2. **Cluster + Global (C+G)**: Combining the two most stable architectures\n\n"
        "We use inverse RMSE weighting: $w_i = \\frac{1/RMSE_i}{\\sum 1/RMSE_j}$"
    )

    new_code = nbf.v4.new_code_cell(
        'print("Calculating Ensemble Predictions...\\n")\n\n'
        '# --- FR ENSEMBLES ---\n'
        'inv_rmse_g_fr = 1.0 / rmse_global_fr\n'
        'inv_rmse_c_fr = 1.0 / rmse_cluster_fr\n'
        'inv_rmse_h_fr = 1.0 / rmse_hourly_fr\n\n'
        '# 1. Hourly + Cluster + Global (FR)\n'
        'w_h_c_g_fr = np.array([inv_rmse_g_fr, inv_rmse_c_fr, inv_rmse_h_fr])\n'
        'w_h_c_g_fr /= w_h_c_g_fr.sum()\n'
        'pred_ens_h_c_g_fr = (w_h_c_g_fr[0] * pred_global_fr + \n'
        '                    w_h_c_g_fr[1] * pred_cluster_fr + \n'
        '                    w_h_c_g_fr[2] * pred_hourly_fr)\n'
        'rmse_ens_h_c_g_fr = np.sqrt(mean_squared_error(y_val_fr, pred_ens_h_c_g_fr))\n\n'
        '# 2. Cluster + Global (FR)\n'
        'w_c_g_fr = np.array([inv_rmse_g_fr, inv_rmse_c_fr])\n'
        'w_c_g_fr /= w_c_g_fr.sum()\n'
        'pred_ens_c_g_fr = (w_c_g_fr[0] * pred_global_fr + \n'
        '                  w_c_g_fr[1] * pred_cluster_fr)\n'
        'rmse_ens_c_g_fr = np.sqrt(mean_squared_error(y_val_fr, pred_ens_c_g_fr))\n\n'
        '# --- UK ENSEMBLES ---\n'
        'inv_rmse_g_uk = 1.0 / rmse_global_uk\n'
        'inv_rmse_c_uk = 1.0 / rmse_cluster_uk\n'
        'inv_rmse_h_uk = 1.0 / rmse_hourly_uk\n\n'
        '# 1. Hourly + Cluster + Global (UK)\n'
        'w_h_c_g_uk = np.array([inv_rmse_g_uk, inv_rmse_c_uk, inv_rmse_h_uk])\n'
        'w_h_c_g_uk /= w_h_c_g_uk.sum()\n'
        'pred_ens_h_c_g_uk = (w_h_c_g_uk[0] * pred_global_uk + \n'
        '                    w_h_c_g_uk[1] * pred_cluster_uk + \n'
        '                    w_h_c_g_uk[2] * pred_hourly_uk)\n'
        'rmse_ens_h_c_g_uk = np.sqrt(mean_squared_error(y_val_uk, pred_ens_h_c_g_uk))\n\n'
        '# 2. Cluster + Global (UK)\n'
        'w_c_g_uk = np.array([inv_rmse_g_uk, inv_rmse_c_uk])\n'
        'w_c_g_uk /= w_c_g_uk.sum()\n'
        'pred_ens_c_g_uk = (w_c_g_uk[0] * pred_global_uk + \n'
        '                  w_c_g_uk[1] * pred_cluster_uk)\n'
        'rmse_ens_c_g_uk = np.sqrt(mean_squared_error(y_val_uk, pred_ens_c_g_uk))\n\n'
        'print(f"FR Ensemble Weights (G, C, H): {w_h_c_g_fr.round(3)}")\n'
        'print(f"FR H+C+G Ensemble RMSE: {rmse_ens_h_c_g_fr:.2f}")\n'
        'print(f"FR C+G Ensemble RMSE:   {rmse_ens_c_g_fr:.2f}")\n'
        'print(f"\\nUK Ensemble Weights (G, C, H): {w_h_c_g_uk.round(3)}")\n'
        'print(f"UK H+C+G Ensemble RMSE: {rmse_ens_h_c_g_uk:.2f}")\n'
        'print(f"UK C+G Ensemble RMSE:   {rmse_ens_c_g_uk:.2f}")'
    )

    # Find where Task 3 starts (the original Task 3: Evaluate and Analyze Residuals)
    # and insert before it.
    idx_task3 = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and 'Task 3: Evaluate and Analyze Residuals' in cell.source:
            idx_task3 = i
            # Rename it to Task 4
            cell.source = cell.source.replace('Task 3', 'Task 4')
            break
    
    if idx_task3 != -1:
        nb.cells.insert(idx_task3, new_markdown)
        nb.cells.insert(idx_task3 + 1, new_code)

    # 2. Update Task 4 (Evaluate and Analyze Residuals) results table
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'results_df = pd.DataFrame({' in cell.source:
            cell.source = cell.source.replace(
                'results_df = pd.DataFrame({\n'
                '    "Market": ["FR", "FR", "FR", "UK", "UK", "UK"],\n'
                '    "Strategy": ["Global", "Cluster (4)", "Hourly (24)", "Global", "Cluster (4)", "Hourly (24)"],\n'
                '    "RMSE": [rmse_global_fr, rmse_cluster_fr, rmse_hourly_fr, \n'
                '             rmse_global_uk, rmse_cluster_uk, rmse_hourly_uk]\n'
                '})',
                'results_df = pd.DataFrame({\n'
                '    "Market": ["FR"]*5 + ["UK"]*5,\n'
                '    "Strategy": ["Global", "Cluster (4)", "Hourly (24)", "H+C+G Ensemble", "C+G Ensemble"] * 2,\n'
                '    "RMSE": [rmse_global_fr, rmse_cluster_fr, rmse_hourly_fr, rmse_ens_h_c_g_fr, rmse_ens_c_g_fr,\n'
                '             rmse_global_uk, rmse_cluster_uk, rmse_hourly_uk, rmse_ens_h_c_g_uk, rmse_ens_c_g_uk]\n'
                '})'
            )
            # Also update improvements display
            cell.source = cell.source.replace(
                'print(f"\\nFR Improvements vs Global:")\n'
                'print(f"  Cluster: {fr_cluster_gain:+.2f}%")\n'
                'print(f"  Hourly:  {fr_hourly_gain:+.2f}%")',
                'print(f"\\nFR Improvements vs Global:")\n'
                'print(f"  Cluster: {fr_cluster_gain:+.2f}%")\n'
                'print(f"  Hourly:  {fr_hourly_gain:+.2f}%")\n'
                'print(f"  H+C+G:   {((rmse_global_fr - rmse_ens_h_c_g_fr) / rmse_global_fr) * 100:+.2f}%")\n'
                'print(f"  C+G:     {((rmse_global_fr - rmse_ens_c_g_fr) / rmse_global_fr) * 100:+.2f}%")'
            )
            cell.source = cell.source.replace(
                'print(f"\\nUK Improvements vs Global:")\n'
                'print(f"  Cluster: {uk_cluster_gain:+.2f}%")\n'
                'print(f"  Hourly:  {uk_hourly_gain:+.2f}%")',
                'print(f"\\nUK Improvements vs Global:")\n'
                'print(f"  Cluster: {uk_cluster_gain:+.2f}%")\n'
                'print(f"  Hourly:  {uk_hourly_gain:+.2f}%")\n'
                'print(f"  H+C+G:   {((rmse_global_uk - rmse_ens_h_c_g_uk) / rmse_global_uk) * 100:+.2f}%")\n'
                'print(f"  C+G:     {((rmse_global_uk - rmse_ens_c_g_uk) / rmse_global_uk) * 100:+.2f}%")'
            )

    # 3. Update Key Findings
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and 'Key Findings' in cell.source:
            # Maybe add a note about ensembles here too if needed
            pass
            
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    add_ensemble_strategies(r'c:\Users\mathi\Code\incommodities-case-crunch-2026\notebooks\04_temporal_segmentation.ipynb')
