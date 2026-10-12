"""
src/utils/taxonomy.py
======================
9-class hierarchical domain taxonomy supporting all 7 datasets.

Taxonomy is aligned to:
  - Our datasets : arXiv, DBLP, Elsevier
  - External     : DBLP-v10, ACL Anthology, S2ORC subset, PMC-OA

Each domain has:
  - A keyword set KW(d) for Eq.(6) keyword scoring
  - An ACM/MAG category alias for external dataset mapping
  - A description for API responses
"""

from __future__ import annotations
from typing import Dict, List, Optional

# ── 9-class taxonomy ──────────────────────────────────────────────────────────
DOMAIN_KWS: Dict[str, List[str]] = {
    "CS_AI_ML": [
        "neural network", "deep learning", "machine learning",
        "transformer", "federated learning", "reinforcement learning",
        "generative model", "attention mechanism", "bert", "gpt",
        "large language model", "llm", "convolutional neural",
        "embedding", "graph neural network", "self-supervised",
        "contrastive learning", "knowledge distillation",
        "object detection", "image classification",
    ],
    "CS_Networks": [
        "wireless network", "vanet", "vehicular network",
        "routing protocol", "antenna design", "channel model",
        "internet of things", "sensor network", "5g network",
        "network latency", "bandwidth allocation", "throughput",
        "network topology", "mac protocol", "heterogeneous network",
        "mobile edge", "network slicing", "spectrum sharing",
    ],
    "CS_Graph_Theory": [
        "graph coloring", "domination number", "vertex coloring",
        "spanning tree", "chromatic number", "clique number",
        "bipartite graph", "graph theory", "automorphism group",
        "planar graph", "hamiltonicity", "independent set",
        "graph isomorphism", "ramsey theory", "extremal graph",
    ],
    "CS_Systems": [
        "distributed system", "cloud computing", "edge computing",
        "parallel computing", "fault tolerant system", "microservice",
        "blockchain", "consensus algorithm", "container orchestration",
        "kubernetes", "operating system", "file system",
        "memory management", "serverless", "stream processing",
    ],
    "Medicine_Biology": [
        "clinical trial", "patient cohort", "disease diagnosis",
        "cancer treatment", "surgical outcome", "organ transplant",
        "gene expression", "protein structure", "drug therapy",
        "epidemiology", "randomized controlled trial",
        "medical imaging", "biomarker", "genomics", "proteomics",
        "cell biology", "immunotherapy", "precision medicine",
    ],
    "Engineering": [
        "tidal energy", "renewable energy", "solar power",
        "wind turbine", "control system", "pid controller",
        "signal processing", "filter design", "power electronics",
        "thermal management", "structural analysis", "fluid dynamics",
        "mechanical design", "vibration analysis", "fatigue",
    ],
    "Mathematics_Physics": [
        "theorem proof", "manifold", "algebraic topology",
        "number theory", "probability distribution",
        "stochastic process", "markov chain", "differential equation",
        "linear algebra", "quantum mechanics", "statistical physics",
        "numerical method", "optimization theory", "combinatorics",
        "functional analysis", "measure theory",
    ],
    "NLP_Linguistics": [
        "natural language processing", "text classification",
        "sentiment analysis", "machine translation",
        "question answering", "information extraction",
        "named entity recognition", "semantic parsing",
        "discourse analysis", "word embedding", "language model",
        "coreference resolution", "summarization", "dialogue system",
    ],
    "Bioinformatics": [
        "sequence alignment", "protein folding", "genome assembly",
        "phylogenetic tree", "rna structure", "dna methylation",
        "metabolic pathway", "systems biology", "computational biology",
        "variant calling", "transcriptomics", "metagenomics",
        "drug discovery", "molecular docking", "structural genomics",
    ],
}

DOMAIN_LIST: List[str] = sorted(DOMAIN_KWS.keys())
N_DOMAINS: int = len(DOMAIN_LIST)

# Human-readable descriptions
DOMAIN_DESC: Dict[str, str] = {
    "CS_AI_ML":            "Artificial Intelligence & Machine Learning",
    "CS_Networks":         "Computer Networks & Communications",
    "CS_Graph_Theory":     "Graph Theory & Combinatorics",
    "CS_Systems":          "Distributed Systems & Cloud",
    "Medicine_Biology":    "Medicine & Biology",
    "Engineering":         "Engineering & Energy Systems",
    "Mathematics_Physics": "Mathematics & Physics",
    "NLP_Linguistics":     "Natural Language Processing",
    "Bioinformatics":      "Bioinformatics & Computational Biology",
}

# External dataset field schemas
DATASET_SCHEMAS: Dict[str, Dict] = {
    # Our 4 datasets (citing_articles schema)
    "arxiv":    {"title": "original_csv_title",
                 "alt_title": "scraped_main_title_from_scholar",
                 "abstract": "original_csv_abstract",
                 "alt_abstract": "scraped_main_abstract_from_scholar",
                 "authors": "scraped_main_authors",
                 "citing_key": "citing_articles",
                 "format": "json_citing"},
    "dblp":     {"title": "original_csv_title",
                 "abstract": "original_csv_abstract",
                 "authors": "scraped_main_authors",
                 "citing_key": "citing_articles",
                 "format": "json_citing"},
    "elsevier": {"title": "scraped_main_title_from_scholar",
                 "abstract": "original_csv_abstract",
                 "authors": "scraped_main_authors",
                 "citing_key": "citing_articles",
                 "format": "json_citing"},
    # External datasets (reference schema)
    "dblp_v10": {"title": "title", "abstract": "abstract",
                 "year": "year", "refs": "references",
                 "format": "jsonl_ref"},
    "acl":      {"title": "title", "abstract": "abstract",
                 "year": "year", "refs": "outCitations",
                 "format": "json_ref"},
    "s2orc":    {"title": "title", "abstract": "abstract",
                 "year": "year", "refs": "outCitations",
                 "format": "jsonl_ref"},
    "pmc":      {"title": "title", "abstract": "abstract",
                 "year": "year", "refs": "references",
                 "format": "jsonl_ref"},
}


def keyword_domain_score(title: str, abstract: str) -> Dict[str, float]:
    """
    Eq.(6): S(d,p) = Σ_{k∈KW(d)} 1[k∈text(p)] / |KW(d)|

    Returns normalized score per domain in [0,1].
    """
    text = (str(title) + " " + str(abstract)).lower()
    return {
        d: sum(1 for k in kws if k in text) / len(kws)
        for d, kws in DOMAIN_KWS.items()
    }


def assign_domain(title: str, abstract: str,
                  min_score: float = 0.02) -> Optional[str]:
    """Return best-matching domain or None if below threshold."""
    scores = keyword_domain_score(title, abstract)
    best   = max(scores, key=scores.get)
    return best if scores[best] >= min_score else None
