"""
src/evaluation/evaluator.py
============================
Three evaluation protocols for the ClassifierAgent.

Protocol 1 — Temporal split
  Train on papers with year ≤ 2022
  Test  on papers with year ≥ 2023
  Simulates real deployment: model trained on past, tested on future.

Protocol 2 — 5-fold Stratified Cross-Validation
  Standard CV with stratification per domain label.
  Reports mean ± std across folds.

Protocol 3 — Leave-One-Source-Out (LOSO)
  Train on 6 datasets, test on the held-out 7th.
  Tests cross-corpus generalization.
  Sources: arxiv, dblp, elsevier, dblp_v10, acl, s2orc, pmc

All protocols report:
  Accuracy · Macro-F1 · Weighted-F1 · Per-class P/R/F1
"""

from __future__ import annotations
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def _metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    rep  = classification_report(y_true, y_pred, output_dict=True,
                                  zero_division=0, digits=4)
    cm   = confusion_matrix(y_true, y_pred,
                             labels=sorted(set(y_true)))
    return {
        "accuracy":         round(acc,  4),
        "macro_f1":         round(mf1,  4),
        "weighted_f1":      round(wf1,  4),
        "per_class":        {k: {kk: round(vv, 4) for kk, vv in v.items()}
                             for k, v in rep.items()
                             if isinstance(v, dict)},
        "confusion_matrix": cm.tolist(),
        "n_samples":        len(y_true),
        "classes":          sorted(set(y_true)),
    }


class Evaluator:
    """
    Runs all three evaluation protocols and writes results to JSON.

    Usage
    -----
    ev = Evaluator(agent, extractor, builder, corpus)
    ev.run_all(output_dir="data/processed/eval")
    """

    def __init__(
        self,
        agent,       # ClassifierAgent
        extractor,   # TDCGFeatureExtractor
        builder,     # TDCGBuilder
        corpus: List[Dict],   # list of {title, abstract, label, source, year, split}
    ):
        self.agent     = agent
        self.extractor = extractor
        self.builder   = builder
        self.corpus    = corpus

    # ── helpers ───────────────────────────────────────────────────────────────
    def _build_clf(self, y=None):
        """
        Returns a fresh _SVCWithProba — no CalibratedClassifierCV,
        no fold-size errors, compatible with every sklearn version.
        """
        return _SVCWithProba(C=1.0)

    def _X_labels(self, subset: Optional[List[int]] = None):
        """Return (X, labels, node_ids) for corpus[subset]."""
        if subset is None:
            ids  = self.extractor.node_ids
            labs = self.extractor.labels
        else:
            ids  = [self.extractor.node_ids[i] for i in subset]
            labs = [self.extractor.labels[i]    for i in subset]

        X, _, _ = self.extractor.transform(self.builder, node_ids=ids)
        return X, labs, ids

    # ── Protocol 1: Temporal split ────────────────────────────────────────────
    def temporal_split(
        self,
        split_year: int = 2023,
    ) -> Dict:
        """
        Train ≤ split_year − 1, Test ≥ split_year.
        """
        logger.info(f"Temporal split (test year ≥ {split_year}) …")
        t0 = time.time()

        tr_idx = [i for i, c in enumerate(self.corpus)
                  if c.get("year") and c["year"] < split_year]
        te_idx = [i for i, c in enumerate(self.corpus)
                  if c.get("year") and c["year"] >= split_year]

        if not tr_idx or not te_idx:
            # Fallback: use 'split' field
            tr_idx = [i for i, c in enumerate(self.corpus) if c.get("split") == "train"]
            te_idx = [i for i, c in enumerate(self.corpus) if c.get("split") == "test"]

        X_tr, y_tr, _ = self._X_labels(tr_idx)
        X_te, y_te, te_ids = self._X_labels(te_idx)

        clf = self._build_clf(y_tr)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        result = _metrics(y_te, preds)
        result.update({
            "protocol":   "temporal_split",
            "split_year": split_year,
            "n_train":    len(tr_idx),
            "n_test":     len(te_idx),
            "elapsed_s":  round(time.time() - t0, 2),
        })
        logger.info(f"Temporal split: acc={result['accuracy']:.4f}  "
                    f"macro-F1={result['macro_f1']:.4f}")
        return result

    # ── Protocol 2: 5-fold CV ─────────────────────────────────────────────────
    def cross_validation(self, n_folds: int = 5) -> Dict:
        """Stratified k-fold cross-validation on full corpus."""
        logger.info(f"{n_folds}-fold stratified CV …")
        t0 = time.time()

        X, y, _ = self._X_labels()
        cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        preds    = cross_val_predict(self._build_clf(y), X, y, cv=cv)

        # Per-fold accuracy
        fold_accs = []
        fold_f1s  = []
        for tr_i, te_i in cv.split(X, y):
            clf = self._build_clf()
            clf.fit(X[tr_i], [y[i] for i in tr_i])
            p = clf.predict(X[te_i])
            fold_accs.append(accuracy_score([y[i] for i in te_i], p))
            fold_f1s.append(f1_score([y[i] for i in te_i], p,
                                      average="macro", zero_division=0))

        result = _metrics(y, preds)
        result.update({
            "protocol":      "cross_validation",
            "n_folds":       n_folds,
            "fold_accuracy": [round(a, 4) for a in fold_accs],
            "fold_macro_f1": [round(f, 4) for f in fold_f1s],
            "acc_mean":      round(float(np.mean(fold_accs)), 4),
            "acc_std":       round(float(np.std(fold_accs)),  4),
            "f1_mean":       round(float(np.mean(fold_f1s)),  4),
            "f1_std":        round(float(np.std(fold_f1s)),   4),
            "elapsed_s":     round(time.time() - t0, 2),
        })
        logger.info(f"5-fold CV: acc={result['acc_mean']:.4f}±{result['acc_std']:.4f}  "
                    f"macro-F1={result['f1_mean']:.4f}±{result['f1_std']:.4f}")
        return result

    # ── Protocol 3: LOSO ──────────────────────────────────────────────────────
    def leave_one_source_out(self) -> Dict:
        """
        For each unique source, train on all other sources,
        test on the held-out source.
        """
        logger.info("Leave-one-source-out evaluation …")
        t0 = time.time()

        sources = sorted(set(self.extractor.sources))
        per_source = {}
        all_true, all_pred = [], []

        for test_src in sources:
            tr_idx = [i for i, s in enumerate(self.extractor.sources) if s != test_src]
            te_idx = [i for i, s in enumerate(self.extractor.sources) if s == test_src]
            if len(te_idx) < 10:
                logger.warning(f"Skipping {test_src}: only {len(te_idx)} samples")
                continue

            X_tr, y_tr, _ = self._X_labels(tr_idx)
            X_te, y_te, _ = self._X_labels(te_idx)

            clf = self._build_clf()
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

            per_source[test_src] = _metrics(y_te, preds)
            per_source[test_src]["n_train"] = len(tr_idx)
            per_source[test_src]["source"]  = test_src
            all_true.extend(y_te)
            all_pred.extend(preds)

            logger.info(f"  {test_src:<14}: "
                        f"acc={per_source[test_src]['accuracy']:.4f}  "
                        f"macro-F1={per_source[test_src]['macro_f1']:.4f}  "
                        f"n_test={len(te_idx)}")

        mean_acc = float(np.mean([v["accuracy"] for v in per_source.values()]))
        mean_f1  = float(np.mean([v["macro_f1"] for v in per_source.values()]))

        result = {
            "protocol":       "leave_one_source_out",
            "per_source":     per_source,
            "mean_accuracy":  round(mean_acc, 4),
            "mean_macro_f1":  round(mean_f1,  4),
            "aggregate":      _metrics(all_true, all_pred) if all_true else {},
            "elapsed_s":      round(time.time() - t0, 2),
        }
        logger.info(f"LOSO mean: acc={mean_acc:.4f}  macro-F1={mean_f1:.4f}")
        return result

    # ── Ensemble ablation ─────────────────────────────────────────────────────
    def ablation_study(self) -> Dict:
        """
        Compare all sub-classifiers and combinations.
        """
        logger.info("Ablation study …")
        t0 = time.time()
        X, y, ids = self._X_labels()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        configs = [
            ("TF-IDF only (no graph)",       False),
            ("TF-IDF + graph features",      True),
        ]
        ablation = {}
        for name, use_graph in configs:
            # re-extract without/with graph
            if use_graph:
                X_abl = X
            else:
                # strip graph columns (last 13 cols)
                X_abl = X[:, :-13]
            preds = cross_val_predict(self._build_clf(y), X_abl, y, cv=cv)
            ablation[name] = _metrics(y, preds)
            ablation[name]["config"] = name

        result = {
            "protocol":  "ablation_study",
            "ablation":  ablation,
            "elapsed_s": round(time.time() - t0, 2),
        }
        for name, m in ablation.items():
            logger.info(f"  {name:<35}: acc={m['accuracy']:.4f}  "
                        f"macro-F1={m['macro_f1']:.4f}")
        return result

    # ── Run all ───────────────────────────────────────────────────────────────
    def run_all(self, output_dir: str = "data/processed/eval") -> Dict:
        """Run all protocols and save results to JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = {}

        results["temporal_split"]        = self.temporal_split()
        results["cross_validation"]      = self.cross_validation()
        results["leave_one_source_out"]  = self.leave_one_source_out()
        results["ablation_study"]        = self.ablation_study()

        # Summary table
        summary = {
            "temporal_split_acc":      results["temporal_split"]["accuracy"],
            "temporal_split_f1":       results["temporal_split"]["macro_f1"],
            "cv_acc_mean":             results["cross_validation"]["acc_mean"],
            "cv_acc_std":              results["cross_validation"]["acc_std"],
            "cv_f1_mean":              results["cross_validation"]["f1_mean"],
            "cv_f1_std":               results["cross_validation"]["f1_std"],
            "loso_mean_acc":           results["leave_one_source_out"]["mean_accuracy"],
            "loso_mean_f1":            results["leave_one_source_out"]["mean_macro_f1"],
        }
        results["summary"] = summary

        out_path = Path(output_dir) / "evaluation_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved → {out_path}")

        # Print summary
        print("\n" + "="*60)
        print("  EVALUATION SUMMARY")
        print("="*60)
        print(f"  Temporal split   : acc={summary['temporal_split_acc']:.4f}  "
              f"F1={summary['temporal_split_f1']:.4f}")
        print(f"  5-fold CV        : acc={summary['cv_acc_mean']:.4f}±{summary['cv_acc_std']:.4f}  "
              f"F1={summary['cv_f1_mean']:.4f}±{summary['cv_f1_std']:.4f}")
        print(f"  LOSO (generalize): acc={summary['loso_mean_acc']:.4f}  "
              f"F1={summary['loso_mean_f1']:.4f}")
        print("="*60 + "\n")
        return results
