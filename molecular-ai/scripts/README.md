# Molecular AI Scripts

This directory contains utility scripts for evaluating and using the models.

## GCPN Molecule Generation and Evaluation

The `evaluate_gcpn.py` script allows you to generate optimized molecules using the trained GCPN model and evaluate their properties.

### Usage

```bash
python evaluate_gcpn.py --vae-model [PATH_TO_VAE_MODEL] --actor-model [PATH_TO_ACTOR_MODEL] [OPTIONS]
```

### Required Arguments

- `--vae-model`: Path to the trained VAE model (e.g., `molecular-ai/models/vae_checkpoints/final_vae_model.pt`)
- `--actor-model`: Path to the trained GCPN actor model (e.g., `molecular-ai/experiments/gcpn_actor_final.pth`)

### Optional Arguments

- `--num-molecules`: Number of molecules to generate (default: 100)
- `--optimize-for`: Property to optimize for, choices: "qed", "logp", "combined" (default: "combined")
- `--output-dir`: Directory to save evaluation results (default: "evaluation_results")

### Example

```bash
# Using local model files
python scripts/evaluate_gcpn.py \
  --vae-model models/vae_checkpoints/final_vae_model.pt \
  --actor-model experiments/gcpn_actor_final.pth \
  --num-molecules 100 \
  --optimize-for combined \
  --output-dir evaluation_results

# Using models from GCS
python scripts/evaluate_gcpn.py \
  --vae-model gs://moleculargeneration-models/vae/final_vae_model.pt \
  --actor-model gs://moleculargeneration-models/gcpn/gcpn_actor_final.pth \
  --num-molecules 100 \
  --optimize-for qed
```

### Output

The script generates the following outputs in the specified output directory:

1. `generated_molecules.csv`: CSV file containing all generated molecules with their SMILES strings and property values
2. `evaluation_metrics.txt`: Text file with evaluation metrics:
   - Validity rate
   - Property statistics (mean, std, min, max)
   - Property improvement
   - Diversity metrics
3. Visualization plots:
   - `qed_distribution.png`: Distribution of QED values
   - `logp_distribution.png`: Distribution of LogP values
   - `qed_logp_scatter.png`: Scatter plot of QED vs LogP
   - `property_improvement.png`: Comparison of property values before and after GCPN optimization
   - `top_molecules.png`: Visualization of top molecules based on reward

### Requirements

The script requires the following libraries:
- torch
- rdkit
- numpy
- pandas
- matplotlib
- seaborn
- tqdm

Additionally, it expects access to the trained VAE and GCPN models.

### Notes

- The script will automatically handle potential errors in the VAE molecule generation, falling back to simple molecules if needed.
- A validation step ensures only chemically valid molecules are included in the final evaluation.
- The script calculates both druglikeness (QED) and lipophilicity (LogP) properties for all generated molecules. 