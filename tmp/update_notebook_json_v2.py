
import json
import os

def update_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Prepare ensemble cells
    ensemble_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Task 3: Ensemble Strategies\n",
            "\n",
            "We now test two ensemble strategies:\n",
            "1. **Hourly + Cluster + Global (H+C+G)**: Combining all three modeling architectures\n",
            "2. **Cluster + Global (C+G)**: Combining the two most stable architectures\n",
            "\n",
            "We use inverse RMSE weighting: $w_i = \\frac{1/RMSE_i}{\\sum 1/RMSE_j}$"
        ]
    }

    ensemble_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Calculating Ensemble Predictions...\\n\")\n",
            "\n",
            "# --- FR ENSEMBLES ---\n",
            "inv_rmse_g_fr = 1.0 / rmse_global_fr\n",
            "inv_rmse_c_fr = 1.0 / rmse_cluster_fr\n",
            "inv_rmse_h_fr = 1.0 / rmse_hourly_fr\n",
            "\n",
            "# 1. Hourly + Cluster + Global (FR)\n",
            "w_h_c_g_fr = np.array([inv_rmse_g_fr, inv_rmse_c_fr, inv_rmse_h_fr])\n",
            "w_h_c_g_fr /= w_h_c_g_fr.sum()\n",
            "pred_ens_h_c_g_fr = (w_h_c_g_fr[0] * pred_global_fr + \n",
            "                    w_h_c_g_fr[1] * pred_cluster_fr + \n",
            "                    w_h_c_g_fr[2] * pred_hourly_fr)\n",
            "rmse_ens_h_c_g_fr = np.sqrt(mean_squared_error(y_val_fr, pred_ens_h_c_g_fr))\n",
            "\n",
            "# 2. Cluster + Global (FR)\n",
            "w_c_g_fr = np.array([inv_rmse_g_fr, inv_rmse_c_fr])\n",
            "w_c_g_fr /= w_c_g_fr.sum()\n",
            "pred_ens_c_g_fr = (w_c_g_fr[0] * pred_global_fr + \n",
            "                  w_c_g_fr[1] * pred_cluster_fr)\n",
            "rmse_ens_c_g_fr = np.sqrt(mean_squared_error(y_val_fr, pred_ens_c_g_fr))\n",
            "\n",
            "# --- UK ENSEMBLES ---\n",
            "inv_rmse_g_uk = 1.0 / rmse_global_uk\n",
            "inv_rmse_c_uk = 1.0 / rmse_cluster_uk\n",
            "inv_rmse_h_uk = 1.0 / rmse_hourly_uk\n",
            "\n",
            "# 1. Hourly + Cluster + Global (UK)\n",
            "w_h_c_g_uk = np.array([inv_rmse_g_uk, inv_rmse_c_uk, inv_rmse_h_uk])\n",
            "w_h_c_g_uk /= w_h_c_g_uk.sum()\n",
            "pred_ens_h_c_g_uk = (w_h_c_g_uk[0] * pred_global_uk + \n",
            "                    w_h_c_g_uk[1] * pred_cluster_uk + \n",
            "                    w_h_c_g_uk[2] * pred_hourly_uk)\n",
            "rmse_ens_h_c_g_uk = np.sqrt(mean_squared_error(y_val_uk, pred_ens_h_c_g_uk))\n",
            "\n",
            "# 2. Cluster + Global (UK)\n",
            "w_c_g_uk = np.array([inv_rmse_g_uk, inv_rmse_c_uk])\n",
            "w_c_g_uk /= w_c_g_uk.sum()\n",
            "pred_ens_c_g_uk = (w_c_g_uk[0] * pred_global_uk + \n",
            "                  w_c_g_uk[1] * pred_cluster_uk)\n",
            "rmse_ens_c_g_uk = np.sqrt(mean_squared_error(y_val_uk, pred_ens_c_g_uk))\n",
            "\n",
            "print(f\"FR Ensemble Weights (G, C, H): {w_h_c_g_fr.round(3)}\")\n",
            "print(f\"FR H+C+G Ensemble RMSE: {rmse_ens_h_c_g_fr:.2f}\")\n",
            "print(f\"FR C+G Ensemble RMSE:   {rmse_ens_c_g_fr:.2f}\")\n",
            "print(f\"\\nUK Ensemble Weights (G, C, H): {w_h_c_g_uk.round(3)}\")\n",
            "print(f\"UK H+C+G Ensemble RMSE: {rmse_ens_h_c_g_uk:.2f}\")\n",
            "print(f\"UK C+G Ensemble RMSE:   {rmse_ens_c_g_uk:.2f}\")"
        ]
    }

    new_cells = []
    task3_inserted = False

    for cell in nb['cells']:
        # Rename Task 3 to Task 4 and insert new Task 3 before it
        if cell['cell_type'] == 'markdown' and any('## Task 3: Evaluate and Analyze Residuals' in s for s in cell['source']):
            if not task3_inserted:
                new_cells.append(ensemble_markdown)
                new_cells.append(ensemble_code)
                task3_inserted = True
            cell['source'] = [s.replace('Task 3', 'Task 4') for s in cell['source']]
        
        # Update results table
        if cell['cell_type'] == 'code' and any('results_df = pd.DataFrame({' in s for s in cell['source']):
            new_source = []
            for line in cell['source']:
                if '"Market": ["FR", "FR", "FR", "UK", "UK", "UK"],' in line:
                    line = line.replace('["FR", "FR", "FR", "UK", "UK", "UK"]', '["FR"]*5 + ["UK"]*5')
                elif '"Strategy": ["Global", "Cluster (4)", "Hourly (24)", "Global", "Cluster (4)", "Hourly (24)"],' in line:
                    line = line.replace('["Global", "Cluster (4)", "Hourly (24)", "Global", "Cluster (4)", "Hourly (24)"]', '["Global", "Cluster (4)", "Hourly (24)", "H+C+G Ensemble", "C+G Ensemble"] * 2')
                elif '"RMSE": [rmse_global_fr, rmse_cluster_fr, rmse_hourly_fr,' in line:
                    line = line.replace('[rmse_global_fr, rmse_cluster_fr, rmse_hourly_fr,', '[rmse_global_fr, rmse_cluster_fr, rmse_hourly_fr, rmse_ens_h_c_g_fr, rmse_ens_c_g_fr,')
                elif 'rmse_global_uk, rmse_cluster_uk, rmse_hourly_uk]' in line:
                    line = line.replace('rmse_global_uk, rmse_cluster_uk, rmse_hourly_uk]', 'rmse_global_uk, rmse_cluster_uk, rmse_hourly_uk, rmse_ens_h_c_g_uk, rmse_ens_c_g_uk]')
                
                # Update improvements
                if 'print(f"  Hourly:  {fr_hourly_gain:+.2f}%")' in line:
                    new_lines = [
                        line,
                        '    print(f"  H+C+G:   {((rmse_global_fr - rmse_ens_h_c_g_fr) / rmse_global_fr) * 100:+.2f}%")\n',
                        '    print(f"  C+G:     {((rmse_global_fr - rmse_ens_c_g_fr) / rmse_global_fr) * 100:+.2f}%")\n'
                    ]
                    new_source.extend(new_lines)
                    continue
                if 'print(f"  Hourly:  {uk_hourly_gain:+.2f}%")' in line:
                    new_lines = [
                        line,
                        '    print(f"  H+C+G:   {((rmse_global_uk - rmse_ens_h_c_g_uk) / rmse_global_uk) * 100:+.2f}%")\n',
                        '    print(f"  C+G:     {((rmse_global_uk - rmse_ens_c_g_uk) / rmse_global_uk) * 100:+.2f}%")\n'
                    ]
                    new_source.extend(new_lines)
                    continue
                new_source.append(line)
            cell['source'] = new_source

        new_cells.append(cell)

    nb['cells'] = new_cells

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    update_notebook(r'c:\Users\mathi\Code\incommodities-case-crunch-2026\notebooks\04_temporal_segmentation.ipynb')
