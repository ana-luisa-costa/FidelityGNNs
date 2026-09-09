## Overview

FidelityGNNs is a systematic benchmark for GNN explanations in PPM based on fidelity. It includes the architecture of a GNN model in src/gnn4ppm, several explanation methods applied for GNNs in src/explainers, fidelity permutation computations in src/fidelity, and benchmarks performed in src/benchmark. This repository supports the publication "Fidelity Benchmark for GNN Explanations in Predictive Process Monitoring".

## Setup

### Prerequisites

- Virtual environment
- Recommended Python version: 3.12

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

Download the data file from: https://figshare.com/s/cde2c35c6ab3f7ca5422, and add the folder to FidelityGNNs.
For reproducibility, we share all information about the datasets BPIC12_A, BPIC12_W, BPIC12_WC, BPIC13_I, BPIC13_O, BPIC17_O, BPIC20_P, and BPIC20_R, as well as embeddings, trained models, evaluation results, fidelity computations, and results obtained in different runs.

### Running the Full Pipeline

Before computing the fidelity values, a GNN model should be trained. This step can be skipped in case the trained model is already provided. As an example for BPIC13_O, a training performed with 30 epochs, 10 hyperparameter epochs, and 30 hyperparameter trials can be performed with the commands:

1. **Train + evaluate** the R-GCN model:

```bash
python3 -m src.gnn4ppm.train \
  --ttl data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --save_best_test data/processed/BPIC13_O/best_val.pt \
  --save_model data/processed/BPIC13_O/best_val_model.pt \
  --vocabs_out data/processed/BPIC13_O/best_val_vocabs.json \
  --epochs 30 --optuna_epochs 10 --trials 30

python3 -m src.gnn4ppm.evaluate \
  --best data/processed/BPIC13_O/best_val.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --output data/processed/BPIC13_O/evaluation.txt
```

2. **Compute fidelity curves** for the trained model, once per heterogeneity mode (`full`, `middle`, `homogeneous`) and 20 sample pairs:

```bash
python3 -m src.fidelity.compute_fidelity_curves \
  --config src/fidelity/configs/BPIC13_O.json \
  --model data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --best-test data/processed/BPIC13_O/best_val.pt \
  --num-samples 20 \
  --heterogeneity-mode full \
  --out data/processed/BPIC13_O/fidelity_curves_full_n20
```

3. **Run the benchmarks**, once fidelity curves exist for the datasets you want to compare:

```bash
python src/benchmark/benchmark_4_targets.py --input data/processed --dir-suffix _n20
python src/benchmark/benchmark_1_overall_score.py --input data/processed --dir-suffix _n20
python src/benchmark/benchmark_2_heterogeneity.py --data-dir data/processed
python src/benchmark/benchmark_3_topk.py --input data/processed
```

`benchmark_4_targets.py` must run before `benchmark_1_overall_score.py`.

### Data Structure

The pipeline expects datasets to be organized as:

```
data/raw/{DATASET}/
├── {DATASET}.ttl              # RDF graph in Turtle format
├── {DATASET}.csv              # Preprocessed event log
├── case_split.json            # train/test case-ID split
├── entity2id.json             # RDF2Vec entity -> node-id mapping
└── entity_embeddings.npy      # pre-built RDF2Vec entity embeddings

data/processed/{DATASET}/
├── best_val.pt                              # test-set predictions
├── best_val_model.pt                        # trained encoder + head checkpoint
├── best_val_vocabs.json                     # class-name vocabularies
├── evaluation.txt                           # prediction-performance report
└── fidelity_curves_{mode}_n{num_samples}/   # per heterogeneity-mode fidelity outputs
    ├── curve_checkpoint.csv
    ├── fidelity_curves_summary.csv
    └── fidelity_curves_auc.csv
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
  --best-test data/processed/BPIC13_O/best_val.pt \
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
  --best-test data/processed/BPIC13_O/best_val.pt \
  --out data/processed/BPIC13_O/explain \
  --num-samples 5
```

#### Integrated Gradients

Uses `captum.attr.IntegratedGradients` against a subgraph-mean baseline.

```bash
python -m src.explainers.explain_ig \
  --ttl  data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb  data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model  data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --best-test data/processed/BPIC13_O/best_val.pt \
  --out data/processed/BPIC13_O/explain \
  --num-samples 5
```

#### PyG GNNExplainer

Learns native node and supplementary edge masks for the model's original predicted class. Ground-truth labels are not required.

```bash
python -m src.explainers.explain_gnnexplainer \
  --ttl data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --out data/processed/BPIC13_O/explain_gnnexplainer \
  --tasks activity,resource,role,lifecycle --num-samples 5
```

#### PGMExplainer

Ranks nodes by `1 - p-value` from chi-square dependence tests against changes in the model's original predicted-class probability. Node perturbations use the complete graph's mean embedding; ground-truth labels are not required.

```bash
python3 -m src.explainers.explain_pgmexplainer \
  --ttl data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --out data/processed/BPIC13_O/explain_pgmexplainer \
  --tasks activity,resource,role,lifecycle --num-samples 5
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
  --best-test data/processed/BPIC13_O/best_val.pt \
  --out data/processed/BPIC13_O/explain \
  --num-samples 5 --shap-candidates 25
```

PROPHET and FOX are two additional explainers provided in this repository.

#### PROPHET explainer

Learns native individual-node and supplementary edge masks for the model's original predicted class. Ground-truth labels are not used by the explainer.

```bash
python -m src.explainers.explain_prophet \
  --ttl  data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb  data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model  data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --out data/processed/BPIC13_O/explain_prophet \
  --dataset BPIC13_O --tasks activity,resource,role,lifecycle --num-samples 5
```
#### FOX neuro-fuzzy surrogate

Trains a small ANFIS (adaptive neuro-fuzzy) surrogate per pair over the top entity candidates, then ranks entities by rule-firing strength. Ground-truth labels are not required.

```bash
python -m src.explainers.explain_fox \
  --ttl  data/raw/BPIC13_O/BPIC13_OpenProblems.ttl \
  --emb  data/raw/BPIC13_O/entity_embeddings.npy \
  --entity2id data/raw/BPIC13_O/entity2id.json \
  --case_split data/raw/BPIC13_O/case_split.json \
  --model  data/processed/BPIC13_O/best_val_model.pt \
  --vocabs data/processed/BPIC13_O/best_val_vocabs.json \
  --best-test data/processed/BPIC13_O/best_val.pt \
  --out data/processed/BPIC13_O/explain \
  --num-samples 5
```
