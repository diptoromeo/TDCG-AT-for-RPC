# TDCG-AT: Temporal Directed Citation Graph with Asymmetric Temporal Weighting

**ClassifierAgent for Research Paper Classification (RPC)**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NAACL 2025](https://img.shields.io/badge/NAACL-2025-red.svg)](https://2025.naacl.org/)

> *Causal Arrows in Citation Graphs: Temporal Directionality as a Structural Prior for Research Paper Classification*

---

## Overview

TDCG-AT is an agentic framework for classifying academic papers into nine
research domains. Rather than treating citation networks as static, undirected
graphs, TDCG-AT enforces the causal arrow of scholarly communication: a paper
can only cite prior work, and recent citations carry stronger domain-proximity
signals than historically distant ones.

The system constructs a **Temporally Directed Citation Graph** (TDCG) in which:

- Every edge `(p_i → p_j)` satisfies `year(p_i) > year(p_j)` — the causal edge constraint
- Each edge carries an asymmetric exponential decay weight `w(e) = exp(−λ·Δt)`
- Graph influence scores (T-PageRank, HITS authority/hub) are derived from this weighted DAG

A **ClassifierAgent** then fuses three complementary sub-classifiers — LinearSVM,
Logistic Regression, and Graph Label Propagation — through weighted soft voting
to produce calibrated domain predictions in under **134 μs per query**.

---


## Key Results

| Dataset | Accuracy | Macro-F₁ |
|---------|----------|----------|
| arXiv | 92.45% | 0.9210 |
| DBLP | 94.15% | 0.9365 |
| Elsevier | 91.85% | 0.9040 |
| PubMed | 95.10% | 0.9445 |
| S2ORC | 91.20% | 0.8965 |
| ACL | 89.75% | 0.8910 |
| **Combined** | **94.35%** | **0.9320** |


Evaluated under 5-fold stratified CV (train 70% / valid 10% / test 20%, seed=42).
Surpasses 15 GNN, temporal-graph, Pre-LM, and ensemble baselines by up to **+6.4 pp**.

---

## Repository Structure

```
NAACL27-(TDCG-AT_ClassifierAgen)/
│
├── main.py                         # Entry point: train / evaluate / serve / demo
├── experiments.py                  # Full experimental suite (all figures & tables)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker image (Python 3.11-slim)
├── docker-compose.yml              # Docker Compose service definition
│
├── data/
│   ├── raw/                        # Own datasets (4 JSON files)
│   │   ├── arXiv_citation_dataset.json
│   │   ├── DBLP_citation_dataset.json
│   │   ├── Elsevier_citation_dataset.json
│   │   └── PubMed_citation_dataset.json
│   ├── external/                   # External datasets (place here)
│   │   ├── acl_anthology.json      # ACL Anthology (73K papers)
│   │   └── s2orc_sample.jsonl      # S2ORC sample (148K papers)
│   └── processed/                  # Auto-generated: graphs, agents, results
│       ├── arXiv/
│       ├── DBLP/
│       ├── Elsevier/
│       ├── PubMed/
│       └── combined/
│
└── src/
    ├── agent/
    │   └── classifier_agent.py     # ClassifierAgent: Perceive→Reason→Act
    ├── graph/
    │   ├── tdcg_builder.py         # Temporal Directed Citation Graph construction
    │   └── feature_extractor.py    # TF-IDF + graph feature extraction
    ├── evaluation/
    │   └── evaluator.py            # CV, temporal split, LOSO, ablation
    ├── api/
    │   └── server.py               # FastAPI REST server
    └── utils/
        └── taxonomy.py             # 9-class domain taxonomy + 117 keywords
```

---

## Domain Taxonomy

TDCG-AT classifies papers into **9 research domains** using a 117-keyword vocabulary:

| ID | Domain | Keywords |
|----|--------|---------|
| d₁ | `CS_AI_ML` | transformer, LLM, federated learning, contrastive learning, … (18) |
| d₂ | `CS_Networks` | VANET, 5G, routing protocol, mobile edge, … (16) |
| d₃ | `CS_Graph_Theory` | chromatic number, domination, hamiltonicity, … (12) |
| d₄ | `CS_Systems` | blockchain, microservice, kubernetes, fault tolerant, … (12) |
| d₅ | `Medicine_Biology` | clinical trial, biomarker, genomics, drug therapy, … (13) |
| d₆ | `Engineering` | renewable energy, PID controller, signal processing, … (12) |
| d₇ | `Mathematics_Physics` | theorem, manifold, stochastic process, quantum mechanics, … (12) |
| d₈ | `NLP_Linguistics` | machine translation, coreference, named entity recognition, … (11) |
| d₉ | `Bioinformatics` | genome assembly, transcriptomics, metagenomics, … (11) |

---

## Installation

### Option 1 — pip (recommended)

```bash
# Clone or extract the repository
cd NAACL27-(TDCG-AT_ClassifierAgen)

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.11+
- scikit-learn ≥ 1.4.0
- networkx ≥ 3.3
- numpy ≥ 1.26.0
- scipy ≥ 1.12.0
- fastapi ≥ 0.111.0
- uvicorn ≥ 0.29.0
- matplotlib ≥ 3.8.0
- seaborn ≥ 0.13.0

### Option 2 — Docker

```bash
# Build and start the API server
docker compose up --build

# The server will be available at http://localhost:8000
```

---

## Data Preparation

### Own Datasets (included)
The four primary datasets are included in `data/raw/`:
- `arXiv_citation_dataset.json` — 867,684 labeled papers (2000–2023)
- `DBLP_citation_dataset.json` — 1,170,826 labeled papers (2002–2023)
- `Elsevier_citation_dataset.json` — 144,686 labeled papers (2015–2023)
- `PubMed_citation_dataset.json` — 418,680 labeled papers (2008–2023)

### External Datasets (download separately)

**ACL Anthology** (~73K papers):
```bash
# Download from HuggingFace
wget https://huggingface.co/datasets/WINGNUS/ACL-OCL/resolve/main/acl-publication-info.74k.parquet

# Convert to JSON (use the provided script)
python fix_acl_parquet.py
# Output: data/external/acl_anthology.json
```

**S2ORC** (~148K papers):
```bash
# Free API key required from: https://www.semanticscholar.org/product/api
# Then run:
python download_s2orc.py --api-key YOUR_KEY --output data/external/s2orc_sample.jsonl
```

---

## Usage

### 1. Train all datasets

```bash
python main.py train
```

This command:
1. Loads each dataset from `data/raw/`
2. Constructs a separate TDCG per dataset (`data/processed/<ds>/tdcg.pkl`)
3. Extracts TF-IDF + graph features (`data/processed/<ds>/extractor.pkl`)
4. Trains and saves a ClassifierAgent per dataset (`data/processed/<ds>/agent.pkl`)
5. Builds a combined pipeline from all 4 own datasets

Expected runtime on a single CPU core:

| Dataset | Build Time |
|---------|-----------|
| arXiv | ~3.0 min |
| DBLP | ~4.0 min |
| Elsevier | ~0.5 min |
| PubMed | ~1.4 min |
| Combined | ~11.9 min |

### 2. Evaluate

```bash
python main.py evaluate
```

Runs 5-fold stratified CV (70/10/20 split), temporal split, LOSO,
and ablation study for all datasets. Results saved to
`data/processed/<ds>/eval_results.json`.

### 3. Run full experiments (all figures and tables)

```bash
python experiments.py
```

Generates all figures (confusion matrices, ablation plots, sensitivity
curves) and saves them to `outputs/`. This reproduces
all results reported in the paper.

### 4. Classify a single paper (demo)

```bash
python main.py demo
```

Classifies 9 sample papers and prints domain predictions with confidence
scores and interdisciplinarity flags.

### 5. Start the REST API server

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up
```

---

## REST API Reference

Once the server is running at `http://localhost:8000`:

### Classify a single paper

```bash
POST /classify
Content-Type: application/json

{
  "title": "Attention Is All You Need",
  "abstract": "We propose a new network architecture, the Transformer...",
  "year": 2017,
  "references": ["Bahdanau et al. 2015", "Hochreiter 1997"]
}
```

**Response:**
```json
{
  "domain": "CS_AI_ML",
  "confidence": 0.9847,
  "all_scores": {
    "CS_AI_ML": 0.9847,
    "NLP_Linguistics": 0.0121,
    ...
  },
  "interdisciplinary": false,
  "inference_time_ms": 0.134
}
```

### Batch classify

```bash
POST /classify/batch
Content-Type: application/json

{
  "papers": [ {...}, {...}, ... ]
}
```

### Health check

```bash
GET /health
```

---

## Reproducing Experiments

To reproduce all paper results from scratch:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Prepare external datasets (see Data Preparation above)

# Step 3: Train all models
python main.py train

# Step 4: Run full experiment suite
python experiments.py

# Step 5: Check outputs/
ls outputs/
```

All random operations use `seed=42` for full reproducibility.
Experiments are deterministic across runs on the same hardware.

---

## Hyperparameters

All values selected by grid search on the combined corpus (5-fold CV, seed=42):

| Component | Parameter | Value |
|-----------|-----------|-------|
| Temporal graph | Decay λ | 0.10 |
| T-PageRank | Teleportation α | 0.15 |
| T-PageRank / HITS | Max iterations | 200 |
| TF-IDF | Vocabulary d_M | 15,000 (own) / 20,000 (combined) |
| TF-IDF | n-gram range | (1, 2) |
| TF-IDF | Title boost | ×3 |
| LinearSVC | C_SVM | 1.0 |
| Logistic Regression | C_LR | 1.0 |
| Label propagation | μ | 0.30 |
| Label propagation | T_max | 15 |
| Label propagation | ε | 10⁻⁴ |
| Ensemble | w_s / w_l / w_g | 0.45 / 0.35 / 0.20 |
| Cross-validation | k | 5 |
| Split ratio | Train/Valid/Test | 70% / 10% / 20% |

---

## Architecture

```
Query paper p
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  PERCEIVE                                           │
│  Extract: x_p ∈ R^d, {S(d,p)}, TPR(p), a(p), B(p) │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  REASON                                             │
│  ├── LinearSVM    → p̂_s ∈ Δ^(K-1)                 │
│  ├── LogReg       → p̂_l ∈ Δ^(K-1)                 │
│  └── GraphProp    → p̂_g ∈ Δ^(K-1)                 │
│                                                     │
│  p̂ = 0.45·p̂_s + 0.35·p̂_l + 0.20·p̂_g            │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│  ACT                                                │
│  d̂ = argmax_d p̂^(d)                               │
│  confidence = max_d p̂^(d)                          │
│  interdisciplinary = |{d : p̂^(d) > 0.15}| ≥ 2     │
└─────────────────────────────────────────────────────┘
```

---

## Citation

If you use TDCG-AT in your research, please cite:

```bibtex
@inproceedings{tdcgat2027,
  title     = {Does Citation Direction Matter? Temporal Directionality as a Structural Prior for Scientific Document Classification},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2025 Annual Conference of the
               North American Chapter of the Association for
               Computational Linguistics (NAACL)},
  year      = {2027},
  note      = {Under review}
}
```

---

## License

This project is released under the **MIT License**.
See `LICENSE` for full terms.

Dataset usage is subject to the terms of each original source:
- arXiv: [arXiv API Terms](https://arxiv.org/help/api/tou)
- DBLP: [DBLP License](https://dblp.org/faq/How+can+I+acknowledge+the+use+of+dblp.html)
- PubMed: [NLM Terms](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html)
- ACL Anthology: [ACL Anthology License](https://aclanthology.org/)
- S2ORC: [S2ORC License](https://github.com/allenai/s2orc)

---

## Contact

For questions about the code or experiments, please open a GitHub issue
or contact the authors via the NAACL submission system.
