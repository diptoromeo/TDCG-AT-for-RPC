"""
src/agent/classifier_agent.py
================================
ClassifierAgent — Perceive → Reason → Act loop

Sub-classifiers
---------------
  LinearSVC with softmax  : no CalibratedClassifierCV — works on any
                            sklearn version and any class size >= 1
  Logistic Regression     : calibrated probability estimates
  GraphPropagation        : exploits citation homophily via TDCG adjacency

Ensemble (Eq. 13)
  p̂ = w_svm · p̂_SVM + w_lr · p̂_LR + w_gp · p̂_GP
  ŷ = argmax_d p̂(d)
"""

from __future__ import annotations
import logging
import pickle
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

from ..utils.taxonomy import (
    DOMAIN_LIST, N_DOMAINS,
    keyword_domain_score, assign_domain,
)

logger = logging.getLogger(__name__)

W_SVM = 0.45
W_LR  = 0.35
W_GP  = 0.20
MU_PROP  = 0.30
MAX_PROP = 15
TOL_PROP = 1e-4


# ── SVC wrapper with softmax probabilities ────────────────────────────────────
# Replaces CalibratedClassifierCV entirely.
# Works on sklearn 1.0 through 1.8+ and any class size >= 1.
# Uses softmax over the SVC decision_function scores — equivalent to
# Platt scaling but requires no cross-validation folds.

class _SVCWithProba(BaseEstimator, ClassifierMixin):
    """
    LinearSVC wrapped with softmax probability estimates.
    Inherits BaseEstimator + ClassifierMixin for full sklearn 1.8+ compatibility.
    Works with cross_val_predict, Pipeline, GridSearchCV.
    No CalibratedClassifierCV — no fold-size errors, any class size >= 1.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C        = C
        self.max_iter = max_iter

    def _make_clf(self):
        return LinearSVC(C=self.C, max_iter=self.max_iter,
                         dual=True, class_weight="balanced")

    def fit(self, X: sp.csr_matrix, y: List[str]) -> "_SVCWithProba":
        self._clf_ = self._make_clf()
        self._clf_.fit(X, y)
        self.classes_ = self._clf_.classes_
        return self

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self._clf_.predict(X)

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        """Softmax over decision function scores → probability matrix."""
        scores = self._clf_.decision_function(X)
        if scores.ndim == 1:
            # Binary case: convert to two-column matrix
            scores = np.column_stack([-scores, scores])
        # Numerically stable softmax
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s  = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)


# ── ClassifierAgent ───────────────────────────────────────────────────────────

class ClassifierAgent:
    """
    Agentic paper domain classifier.

    Perceive : observe X = [TF-IDF | graph features]
    Reason   : LinearSVC + LogisticRegression + GraphPropagation
    Act      : emit {domain, confidence, all_scores, is_interdisciplinary}
    """

    def __init__(
        self,
        C_svm:         float = 1.0,
        C_lr:          float = 1.0,
        mu:            float = MU_PROP,
        max_iter_prop: int   = MAX_PROP,
        n_cv_folds:    int   = 3,          # kept for API compatibility
    ):
        self.C_svm         = C_svm
        self.C_lr          = C_lr
        self.mu            = mu
        self.max_iter_prop = max_iter_prop
        self.n_cv_folds    = n_cv_folds

        self._svm: Optional[_SVCWithProba]       = None
        self._lr:  Optional[LogisticRegression]  = None
        self.classes_:    List[str] = []
        self._cls_idx:    Dict[str, int] = {}
        self._fitted      = False
        self.training_stats: Dict = {}

    # ── PERCEIVE ──────────────────────────────────────────────────────────────
    def _perceive(self, X: sp.csr_matrix,
                  labels: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
        assert X.shape[0] == len(labels), \
            f"X rows {X.shape[0]} != labels {len(labels)}"
        logger.info(
            f"Perceive: {X.shape[0]} papers, "
            f"{X.shape[1]} features, "
            f"{len(set(labels))} classes"
        )
        return X, labels

    # ── REASON — FIT ──────────────────────────────────────────────────────────
    def fit(self, X: sp.csr_matrix, labels: List[str]) -> "ClassifierAgent":
        X, labels = self._perceive(X, labels)
        self.classes_ = sorted(set(labels))
        self._cls_idx = {c: i for i, c in enumerate(self.classes_)}

        # ── LinearSVC with softmax (no CalibratedClassifierCV) ───────────
        # _SVCWithProba works for ANY class size and ANY sklearn version
        # because it never calls CalibratedClassifierCV internally.
        logger.info("Fitting LinearSVC with softmax probabilities ...")
        self._svm = _SVCWithProba(C=self.C_svm)
        self._svm.fit(X, labels)

        # ── Logistic Regression ──────────────────────────────────────────
        logger.info("Fitting Logistic Regression ...")
        self._lr = LogisticRegression(
            C=self.C_lr,
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
        )
        self._lr.fit(X, labels)

        self._fitted = True
        logger.info("ClassifierAgent fitted successfully.")
        return self

    # ── REASON — GRAPH PROPAGATION ────────────────────────────────────────────
    def _graph_propagation(
        self,
        X:        sp.csr_matrix,
        node_ids: List[str],
        builder,
    ) -> np.ndarray:
        """
        Eq.(12): Y^{t+1} = (1-mu)*Y0 + mu*A_norm*Y^t
        Seeds from SVM posterior. Converges by Banach contraction (mu<1).
        """
        if not self._fitted:
            raise RuntimeError("Fit before calling propagation.")

        G     = builder.G
        n     = len(node_ids)
        n_cls = len(self.classes_)

        # Seed from SVM softmax probabilities
        Y0 = self._svm.predict_proba(X)
        svm_cls = list(self._svm.classes_)
        reorder = [svm_cls.index(c) if c in svm_cls else 0
                   for c in self.classes_]
        Y0 = Y0[:, reorder]
        Y  = Y0.copy()

        # Build sparse symmetric adjacency from TDCG edge weights
        rows, cols, wts = [], [], []
        nid2idx = {nid: i for i, nid in enumerate(node_ids)}
        for u, v, edata in G.edges(data=True):
            ui = nid2idx.get(u, -1)
            vi = nid2idx.get(v, -1)
            if ui >= 0 and vi >= 0:
                w = edata.get("weight", 1.0)
                rows += [ui, vi]
                cols += [vi, ui]
                wts  += [w,   w]

        if not rows:
            return Y0

        A = sp.csr_matrix(
            (wts, (rows, cols)), shape=(n, n), dtype=np.float32
        )
        row_sum = np.array(A.sum(axis=1)).flatten() + 1e-12
        A_norm  = sp.diags(1.0 / row_sum) @ A

        for it in range(self.max_iter_prop):
            Y_new = (1 - self.mu) * Y0 + self.mu * (A_norm @ Y)
            delta = float(np.abs(Y_new - Y).max())
            Y     = Y_new
            if delta < TOL_PROP:
                logger.debug(f"GraphProp converged at iter {it + 1}")
                break

        return Y

    # ── ACT — PREDICT ─────────────────────────────────────────────────────────
    def predict(
        self,
        X:        sp.csr_matrix,
        node_ids: Optional[List[str]] = None,
        builder=None,
    ) -> np.ndarray:
        proba = self.predict_proba(X, node_ids, builder)
        return np.array(self.classes_)[proba.argmax(axis=1)]

    def predict_proba(
        self,
        X:        sp.csr_matrix,
        node_ids: Optional[List[str]] = None,
        builder=None,
    ) -> np.ndarray:
        """
        Eq.(13): p̂ = W_SVM*p̂_SVM + W_LR*p̂_LR + W_GP*p̂_GP
        """
        if not self._fitted:
            raise RuntimeError("Fit before predict.")

        p_svm = self._align(self._svm.predict_proba(X),
                             list(self._svm.classes_))
        p_lr  = self._align(self._lr.predict_proba(X),
                             list(self._lr.classes_))

        if node_ids and builder is not None:
            p_gp = self._graph_propagation(X, node_ids, builder)
            p_gp = self._align(p_gp, self.classes_)
        else:
            p_gp = p_svm

        return W_SVM * p_svm + W_LR * p_lr + W_GP * p_gp

    def _align(self, proba: np.ndarray,
               clf_classes: List[str]) -> np.ndarray:
        aligned = np.zeros(
            (proba.shape[0], len(self.classes_)), dtype=np.float32
        )
        for j, cls in enumerate(clf_classes):
            if cls in self._cls_idx:
                aligned[:, self._cls_idx[cls]] = proba[:, j]
        return aligned

    # ── CLASSIFY SINGLE PAPER ─────────────────────────────────────────────────
    def classify_paper(
        self,
        title:    str,
        abstract: str  = "",
        extractor=None,
        builder=None,
        node_id:  Optional[str] = None,
    ) -> Dict:
        if extractor is None:
            raise ValueError("extractor required.")

        dom_scores = keyword_domain_score(title, abstract)
        graph_vec  = np.array(
            [0.0, 0.0, 0.0, 0.0] +
            [dom_scores.get(d, 0.0) for d in DOMAIN_LIST],
            dtype=np.float32,
        )

        X   = extractor.transform_text(title, abstract, graph_vec)
        p   = self.predict_proba(X)
        idx = int(p.argmax())

        domain = self.classes_[idx]
        conf   = float(p[0, idx])
        all_sc = {c: round(float(p[0, i]), 5)
                  for i, c in enumerate(self.classes_)}

        kw_best = max(dom_scores, key=dom_scores.get)
        kw_dom  = kw_best if dom_scores[kw_best] >= 0.02 else "Unknown"
        inter   = sum(1 for v in p[0] if v >= 0.15) >= 2

        return {
            "domain":               domain,
            "confidence":           round(conf, 4),
            "all_scores":           all_sc,
            "keyword_domain":       kw_dom,
            "is_interdisciplinary": inter,
        }

    # ── PERSISTENCE ───────────────────────────────────────────────────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "svm":     self._svm,
                "lr":      self._lr,
                "classes": self.classes_,
                "cls_idx": self._cls_idx,
                "training_stats": self.training_stats,
                "config": {
                    "C_svm": self.C_svm,
                    "C_lr":  self.C_lr,
                    "mu":    self.mu,
                    "max_iter_prop": self.max_iter_prop,
                },
            }, f, protocol=4)
        logger.info(f"ClassifierAgent saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ClassifierAgent":
        with open(path, "rb") as f:
            d = pickle.load(f)
        cfg = d.get("config", {})
        obj = cls(
            C_svm=cfg.get("C_svm", 1.0),
            C_lr=cfg.get("C_lr",   1.0),
            mu=cfg.get("mu",       MU_PROP),
            max_iter_prop=cfg.get("max_iter_prop", MAX_PROP),
        )
        obj._svm           = d["svm"]
        obj._lr            = d["lr"]
        obj.classes_       = d["classes"]
        obj._cls_idx       = d["cls_idx"]
        obj.training_stats = d.get("training_stats", {})
        obj._fitted        = True
        return obj
