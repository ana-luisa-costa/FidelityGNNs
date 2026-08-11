## Overview

GNN4PPM is a pipeline that combines RDF graph embeddings with relational graph neural networks to predict process attributes and supports the publication "GNN4PPM: Multi-Target Predictive Process Monitoring with Relational Graph Convolutional Networks".

## Setup

### Prerequisites

- Virtual environment
- Recommended Python version: 3.9.6

### Installation

1. Create and activate the virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data

Download the data file from: https://figshare.com/s/cde2c35c6ab3f7ca5422, and add the folder to GNN4PPM.
For reproducibility, we share all information about the datasets BPIC12_A, BPIC12_W, BPIC12_WC, BPIC13_O, BPIC17_O, BPIC20_P, and BPIC20_R.

### Running the Full Pipeline

Run `main.py` for a specified dataset, such as for BPIC13_O.

```bash
python main.py BPIC13_O
```

### KG Construction (YARRRML -> RML -> RDF)

RDF graphs can be generated from the dataset-specific YARRRML mappings in `kg-construction/`.

- Full instructions: `kg-construction/README.md`
- Includes workflow for:
  - loading CSV files into PostgreSQL,
  - parsing YARRRML into RML,
  - executing RMLMapper to generate `.ttl` RDF files.
- Dataset folders also include named helper scripts (`.sh` and `.bat`) to run parsing and mapping in one step.


### Data Structure

The pipeline expects datasets to be organized as:

```
data/raw/{DATASET}/
├── {DATASET}.ttl          # RDF graph in Turtle format
├── {DATASET}.csv          # Preprocessed event log
└── (generated files after build)
    ├── case_split.json
    ├── entity2id.json
    └── entity_embeddings.npy

data/processed/{DATASET}/
└── (generated files after pipeline)
    ├── best_val.pt
    ├── best_val_vocabs.json
    └── evaluation.txt
```



### Explainability

After training, several per-pair explainability methods can be run independently.  All the results will be under `data/processed/{DATASET}/explain/` (or a custom `--out` path).

#### GraphLIME neighbor mask (with Ridge)

```bash
python -m src.explainers.explain_lime \
  --ttl  data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb  data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model  data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --best-val data/processed/BPIC13_O/best_val.pt \
  --out data/processed/BPIC13_O/explain \
  --num-samples 5
```

#### Gradient saliency

```bash
python -m src.explainers.explain_gradient \
  --ttl  data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb  data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model  data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --best-val data/processed/BPIC13_O/best_val.pt \
  --out data/processed/BPIC13_O/explain \
  --num-samples 5
```

#### PROPHET explainer

Learns native individual-node and supplementary edge masks for the model's
original predicted class. Ground-truth labels are not used by the explainer.

```bash
python -m src.explainers.explain_prophet \
  --ttl  data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb  data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model  data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --out data/processed/BPIC13_O/explain_prophet \
  --dataset BPIC13_O --tasks activity,resource,role,lifecycle --num-samples 1
```

#### PyG GNNExplainer

Learns native node and supplementary edge masks for the model's original
predicted class. Ground-truth labels are not required.

```bash
python -m src.explainers.explain_gnnexplainer \
  --ttl data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --out data/processed/BPIC13_O/explain_gnnexplainer \
  --tasks activity,resource,role,lifecycle --num-samples 1
```

#### PGMExplainer

Ranks nodes by `1 - p-value` from chi-square dependence tests against changes
in the model's original predicted-class probability. Node perturbations use the
complete graph's mean embedding; ground-truth labels are not required.

```bash
python3 -m src.explainers.explain_pgmexplainer \
  --ttl data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --out data/processed/BPIC13_O/explain_pgmexplainer \
  --tasks activity,resource,role,lifecycle --num-samples 1
```

#### KernelSHAP entity-level explainer

Uses `shap.KernelExplainer`.

```bash
python -m src.explainers.explain_shap \
  --ttl  data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb  data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model  data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --best-val data/processed/BPIC13_O/best_val.pt \
  --out data/processed/BPIC13_O/explain \
  --num-samples 5 --shap-candidates 25
```


Compute signed probability fidelity locally (default). Probability Fidelity+
and Fidelity- are the original predicted-class probability minus the respective
perturbed probability:

```bash
python3 -m src.fidelity.compute_fidelity_curves \
  --config src/fidelity/configs/BPIC13_O.json \
  --out data/processed/BPIC13_O/fidelity_prob \
  --fidelity-metrics prob
```

The categorical fidelity task aliases are `activity`, `resource`, `role`, and
`lifecycle`. Missing aliases are skipped per dataset, so BPIC12 runs skip
`role`, BPIC20 runs skip `lifecycle`, and BPIC13_O runs all four. BPIC12
`org_resource` is treated as categorical even though the raw column is named
`event_otherN_org_resource`.

Use `--fidelity-metrics acc` for class-change fidelity or
`--fidelity-metrics acc,prob` for both. Use a fresh output directory when
changing metric modes because the checkpoint schema is fixed and completed-row
keys are shared. Outputs created before the signed probability-drop definition
must also use a fresh directory.

Run three-seed parallel fidelity on COMA

```
SEEDS=42,43,44 SAMPLE_SEED=42 BASE_DIR=plots_parallel \
bash scripts/submit_fidelity_parallel.sh
```

Merged curves show mean fidelity across all evaluated pair-seed runs.
Bands show SEM across pair-seed runs. For deterministic methods, this reflects
pair-to-pair variability. For stochastic methods, this reflects both
pair-to-pair variability and seed-driven randomness.

Run plot

```
MPLCONFIGDIR=/private/tmp/gnn4ppm_mpl_cache \
python3 -m src.fidelity.plot_fidelity_curves \
  --summary plots_parallel/merged/fidelity_curves_summary.csv \
  --out plots_parallel/merged \
  --formats png
```
