#!/usr/bin/env python3
"""
main.py — TDCG-AT ClassifierAgent System
==========================================
Commands
--------
  python main.py train       build graph → fit agent → save
  python main.py evaluate    run all 3 evaluation protocols
  python main.py serve       start FastAPI server
  python main.py demo        classify 9 sample papers

Datasets supported
------------------
  Our 3  : arXiv, DBLP, Elsevier  (data/raw/)
  External (place in data/external/):
    DBLP-v10      → dblp_v10.jsonl
    ACL Anthology → acl_anthology.json
    S2ORC subset  → s2orc_subset.jsonl
    PMC-OA        → pmc_oa.jsonl
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_RAW  = "data/raw"
DATA_PROC = "data/processed"
DATA_EXT  = "data/external"


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_or_build_pipeline(
    force_rebuild: bool = False,
    max_ext_papers: int = 200_000,
):
    from src.graph.tdcg_builder      import TDCGBuilder
    from src.graph.feature_extractor import TDCGFeatureExtractor
    from src.agent.classifier_agent  import ClassifierAgent

    graph_pkl     = os.path.join(DATA_PROC, "tdcg.pkl")
    agent_pkl     = os.path.join(DATA_PROC, "agent.pkl")
    extractor_pkl = os.path.join(DATA_PROC, "extractor.pkl")
    Path(DATA_PROC).mkdir(parents=True, exist_ok=True)

    # ── graph ────────────────────────────────────────────────────────────────
    if not force_rebuild and Path(graph_pkl).exists():
        logger.info("Loading cached graph …")
        builder = TDCGBuilder.load(graph_pkl)
    else:
        logger.info("Building TDCG from all datasets …")
        builder = TDCGBuilder()

        for fname, schema in [
            ("arXiv_citation_dataset.json",    "arxiv"),
            ("DBLP_citation_dataset.json",     "dblp"),
            ("Elsevier_citation_dataset.json", "elsevier"),
        ]:
            p = os.path.join(DATA_RAW, fname)
            if Path(p).exists():
                builder.load_json_citing(p, schema)
            else:
                logger.warning(f"Not found: {p}")

        ext_map = {
            "dblp_v10.jsonl":     "dblp_v10",
            "s2orc_subset.jsonl": "s2orc",
            "pmc_oa.jsonl":       "pmc",
        }
        jsonl_exts = set(ext_map.keys())
        for fname, schema in ext_map.items():
            p = os.path.join(DATA_EXT, fname)
            if Path(p).exists():
                logger.info(f"Loading {fname} (max {max_ext_papers}) …")
                builder.load_jsonl_ref(p, schema, max_papers=max_ext_papers)

        acl_path = os.path.join(DATA_EXT, "acl_anthology.json")
        if Path(acl_path).exists():
            builder.load_json_ref(acl_path, "acl")

        builder.finalize()
        builder.save(graph_pkl)

    # ── extractor + agent ─────────────────────────────────────────────────────
    if (not force_rebuild
            and Path(extractor_pkl).exists()
            and Path(agent_pkl).exists()):
        logger.info("Loading cached extractor + agent …")
        with open(extractor_pkl, "rb") as f:
            extractor = pickle.load(f)
        agent = ClassifierAgent.load(agent_pkl)
    else:
        logger.info("Fitting feature extractor …")
        extractor = TDCGFeatureExtractor()
        extractor.fit(builder)
        X, node_ids, labels = extractor.transform(builder)

        logger.info(f"Feature matrix: {X.shape}")
        logger.info(f"Labels: {len(set(labels))} classes, {len(labels)} samples")

        logger.info("Fitting ClassifierAgent …")
        # Drop classes that have only 1 sample — LogisticRegression still
        # needs at least 2 distinct class labels to train.  Any class with
        # >= 2 samples is handled by the adaptive calibration in fit().
        from collections import Counter as _Ctr
        label_counts = _Ctr(labels)
        singleton_classes = [c for c, n in label_counts.items() if n < 2]
        if singleton_classes:
            logger.warning(
                f"Removing {len(singleton_classes)} singleton class(es) "
                f"that cannot be trained: {singleton_classes}"
            )
            valid_idx = [i for i, lb in enumerate(labels)
                         if label_counts[lb] >= 2]
            X      = X[valid_idx]
            labels = [labels[i] for i in valid_idx]
            logger.info(
                f"Corpus after removal: {len(labels)} samples, "
                f"{len(set(labels))} classes."
            )

        agent = ClassifierAgent()
        agent.fit(X, labels)

        with open(extractor_pkl, "wb") as f:
            pickle.dump(extractor, f, protocol=4)
        agent.save(agent_pkl)

    return builder, extractor, agent


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def cmd_train(args):
    builder, extractor, agent = _load_or_build_pipeline(
        force_rebuild=args.rebuild,
        max_ext_papers=args.max_ext,
    )
    st = builder.stats
    print(f"\n{'='*60}")
    print(f"  TDCG-AT ClassifierAgent — trained")
    print(f"{'='*60}")
    print(f"  Graph nodes  : {st.get('n_nodes',0):,}")
    print(f"  Graph edges  : {st.get('n_edges',0):,}")
    print(f"  WCC clusters : {st.get('n_wcc',0):,}")
    print(f"  Labeled samples: {extractor.n_samples:,}")
    print(f"  Feature dim    : {extractor.n_features:,}")
    print(f"  Classes        : {agent.classes_}")
    print(f"{'='*60}\n")


# ── EVALUATE ──────────────────────────────────────────────────────────────────
def cmd_evaluate(args):
    from src.evaluation.evaluator import Evaluator

    builder, extractor, agent = _load_or_build_pipeline()

    corpus = [
        {"nid": nid, "label": lab, "source": src, "year": yr,
         "split": "test" if (yr and yr >= 2023) else "train"}
        for nid, lab, src, yr in zip(
            extractor.node_ids, extractor.labels,
            extractor.sources, extractor.years,
        )
    ]

    ev = Evaluator(agent, extractor, builder, corpus)

    proto = args.protocol
    if proto == "all":
        ev.run_all(output_dir=DATA_PROC + "/eval")
    elif proto == "temporal":
        r = ev.temporal_split()
        print(f"Temporal split: acc={r['accuracy']}  macro-F1={r['macro_f1']}")
    elif proto == "cv":
        r = ev.cross_validation()
        print(f"5-fold CV: acc={r['acc_mean']}±{r['acc_std']}  "
              f"F1={r['f1_mean']}±{r['f1_std']}")
    elif proto == "loso":
        r = ev.leave_one_source_out()
        print(f"LOSO: acc={r['mean_accuracy']}  F1={r['mean_macro_f1']}")
    elif proto == "ablation":
        r = ev.ablation_study()
        for name, m in r["ablation"].items():
            print(f"{name}: acc={m['accuracy']}  F1={m['macro_f1']}")


# ── SERVE ─────────────────────────────────────────────────────────────────────
def cmd_serve(args):
    import uvicorn
    os.environ["DATA_DIR"]  = DATA_RAW
    os.environ["PROC_DIR"]  = DATA_PROC
    os.environ["EXT_DIR"]   = DATA_EXT
    os.environ["MAX_EXT_PAPERS"] = str(args.max_ext)
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


# ── DEMO ──────────────────────────────────────────────────────────────────────
def cmd_demo(args):
    builder, extractor, agent = _load_or_build_pipeline()

    DEMO_PAPERS = [
        ("Federated Learning with Differential Privacy for IoT",
         "We propose a novel federated learning framework combining differential privacy "
         "with edge computing for neural network training on distributed IoT devices.",
         "CS_AI_ML"),
        ("Tidal Turbine Control with Adaptive PID",
         "An adaptive PID controller for tidal energy harvesting turbines that adjusts "
         "to varying tidal flow, improving efficiency by 23% over fixed controllers.",
         "Engineering"),
        ("On the Chromatic Number of Hajos Sum of Graphs",
         "We establish new bounds on the chromatic polynomial of the Hajos sum using "
         "algebraic methods and study domination in vertex coloring operations.",
         "CS_Graph_Theory"),
        ("Clinical Outcomes of Liver Transplantation in Pediatric Patients",
         "A retrospective cohort study of 347 pediatric liver transplant recipients. "
         "Primary endpoints include graft survival and post-operative complications.",
         "Medicine_Biology"),
        ("VANET Routing Protocol with 5G Channel Model Integration",
         "Vehicular ad-hoc network routing enhanced with 5G millimeter wave channel "
         "models. Reduces latency by 41% in urban scenarios.",
         "CS_Networks"),
        ("Stochastic Differential Equations on Riemannian Manifolds",
         "We prove a generalization of Ito's lemma to compact Riemannian manifolds "
         "and establish convergence rates for the corresponding Markov chains.",
         "Mathematics_Physics"),
        ("GPT-4 Fine-tuning for Scientific Literature Summarization",
         "Domain-specific fine-tuning of large language models on 50,000 academic papers "
         "improves ROUGE-L by 18% compared to zero-shot GPT-4.",
         "CS_AI_ML"),
        ("Named Entity Recognition for Biomedical Text Mining",
         "Transformer-based NLP model for extracting biomedical entities from clinical "
         "notes using transfer learning and coreference resolution.",
         "NLP_Linguistics"),
        ("Protein Structure Prediction via Sequence Alignment",
         "A novel sequence alignment algorithm for protein folding prediction using "
         "phylogenetic tree analysis and genome assembly techniques.",
         "Bioinformatics"),
    ]

    print(f"\n{'='*70}")
    print(f"  TDCG-AT ClassifierAgent — DEMO  ({len(DEMO_PAPERS)} papers)")
    print(f"{'='*70}")

    correct = 0
    for i, (title, abstract, true_label) in enumerate(DEMO_PAPERS, 1):
        result = agent.classify_paper(title, abstract, extractor, builder)
        pred   = result["domain"]
        conf   = result["confidence"]
        match  = "✓" if pred == true_label else "✗"
        if pred == true_label:
            correct += 1

        print(f"\n  [{i}] {title[:62]}")
        print(f"       True : {true_label:<24}  Pred : {pred:<24}  {match}  ({conf:.3f})")
        top3 = sorted(result["all_scores"].items(), key=lambda x: -x[1])[:3]
        for d, s in top3:
            bar = "█" * int(s * 20)
            print(f"         {d:<24}  {s:.4f}  {bar}")

    print(f"\n  Demo accuracy: {correct}/{len(DEMO_PAPERS)} = {100*correct/len(DEMO_PAPERS):.1f}%")

    # Graph stats
    st = builder.stats
    print(f"\n  Graph: {st.get('n_nodes',0):,} nodes · {st.get('n_edges',0):,} edges"
          f" · λ={st.get('lambda_decay',0.10)}")
    print(f"  Feature matrix: {extractor.n_samples:,} × {extractor.n_features:,}")
    print(f"\n{'='*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TDCG-AT ClassifierAgent System")
    p.add_argument("command", nargs="?", default="demo",
                   choices=["train", "evaluate", "serve", "demo"])
    p.add_argument("--rebuild",   action="store_true", help="Force rebuild graph+agent")
    p.add_argument("--max-ext",   type=int, default=200_000, dest="max_ext",
                   help="Max papers to load from each external JSONL dataset")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=8000)
    p.add_argument("--reload",    action="store_true")
    p.add_argument("--protocol",  default="all",
                   choices=["all","temporal","cv","loso","ablation"],
                   help="Evaluation protocol (for evaluate command)")

    args = p.parse_args()
    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(DATA_EXT,  exist_ok=True)

    if   args.command == "train":    cmd_train(args)
    elif args.command == "evaluate": cmd_evaluate(args)
    elif args.command == "serve":    cmd_serve(args)
    else:                            cmd_demo(args)
