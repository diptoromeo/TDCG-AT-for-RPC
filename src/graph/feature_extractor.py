"""
src/graph/feature_extractor.py
================================
Builds joint feature matrix X = [X_TF-IDF | X_graph] ∈ ℝ^{N×(d+13)}

X_TF-IDF  : (N, 20000)  TF-IDF with title×3 boost
X_graph   : (N, 13)     [in_deg, out_deg, tpr, hub, auth, cluster_size,
                          dom_0, dom_1, …, dom_8]  (9 domain scores)

Equations referenced:
  Eq.(8)  title boosting:  text*(p) = title∥title∥title∥abstract
  Eq.(9)  graph vector:    g_v = [in_deg_norm, …, S(d_1,p), …, S(d_9,p)]
  Eq.(10) joint matrix:    X = [M | G]
"""

from __future__ import annotations
import logging
import numpy as np
import scipy.sparse as sp
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .tdcg_builder import TDCGBuilder
from ..utils.taxonomy import DOMAIN_LIST, N_DOMAINS

logger = logging.getLogger(__name__)

GRAPH_DIM = 4 + N_DOMAINS   # in_deg, out_deg, tpr, auth + 9 domain scores = 13


class TDCGFeatureExtractor:
    """
    Fits a TF-IDF vectorizer and builds graph feature columns.

    Parameters
    ----------
    max_features   : TF-IDF vocabulary size
    ngram_range    : n-gram range
    title_boost    : how many times to repeat title before abstract
    min_df         : minimum document frequency for TF-IDF terms
    use_graph      : whether to append graph features
    """

    def __init__(
        self,
        max_features:  int  = 20_000,
        ngram_range:   Tuple[int, int] = (1, 2),
        title_boost:   int  = 3,
        min_df:        int  = 2,
        sublinear_tf:  bool = True,
        use_graph:     bool = True,
    ):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.title_boost  = title_boost
        self.min_df       = min_df
        self.sublinear_tf = sublinear_tf
        self.use_graph    = use_graph

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.node_ids:   List[str] = []
        self.labels:     List[str] = []
        self.sources:    List[str] = []
        self.years:      List[Optional[int]] = []
        self._fitted     = False

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, builder: TDCGBuilder,
            label_key: str = "domain",
            min_label_score: float = 0.02) -> "TDCGFeatureExtractor":
        """
        Fit TF-IDF on all labeled nodes in the graph.
        Only nodes with domain != 'Unknown' are included.
        """
        G = builder.G
        self.node_ids = []
        self.labels   = []
        self.sources  = []
        self.years    = []
        texts         = []

        for nid, d in G.nodes(data=True):
            label  = d.get(label_key, "Unknown")
            if label == "Unknown":
                continue
            title    = d.get("title",    "")
            abstract = d.get("abstract", "")
            # Eq.(8): title × boost
            combined = (f"{title} " * self.title_boost + abstract).strip()
            texts.append(combined or "empty")
            self.node_ids.append(nid)
            self.labels.append(label)
            self.sources.append(d.get("source", ""))
            self.years.append(d.get("year"))

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            strip_accents="unicode",
        )
        self.vectorizer.fit(texts)
        self._fitted = True

        logger.info(f"Fitted extractor: {len(self.node_ids)} labeled nodes, "
                    f"{self.vectorizer.vocabulary_.__len__()} TF-IDF features")
        return self

    # ── transform ─────────────────────────────────────────────────────────────
    def transform(
        self,
        builder: TDCGBuilder,
        node_ids: Optional[List[str]] = None,
    ) -> Tuple[sp.csr_matrix, List[str], List[str]]:
        """
        Build feature matrix.

        Returns
        -------
        X       : sp.csr_matrix, shape (n, d + GRAPH_DIM)
        nids    : list of node IDs in row order
        labels  : list of domain labels in row order
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        G    = builder.G
        ids  = node_ids or self.node_ids
        labs = [G.nodes[n].get("domain", "Unknown") for n in ids]

        # TF-IDF block
        texts = []
        for n in ids:
            d = G.nodes.get(n, {})
            combined = (f"{d.get('title','')} " * self.title_boost
                        + d.get("abstract", "")).strip()
            texts.append(combined or "empty")
        X_tf = self.vectorizer.transform(texts)

        if not self.use_graph:
            return X_tf, ids, labs

        # Graph feature block — Eq.(9)
        G_feat = np.zeros((len(ids), GRAPH_DIM), dtype=np.float32)
        for i, n in enumerate(ids):
            d = G.nodes.get(n, {})
            G_feat[i, 0] = d.get("in_deg_norm",    0.0)
            G_feat[i, 1] = d.get("out_deg_norm",   0.0)
            G_feat[i, 2] = d.get("tpr_norm",       0.0) * 10
            G_feat[i, 3] = d.get("auth_norm",      0.0)
            for j, dom in enumerate(DOMAIN_LIST):
                G_feat[i, 4 + j] = d.get(f"dom_{dom}", 0.0)

        # Normalize graph columns to [0,1]
        col_max = G_feat.max(axis=0) + 1e-12
        G_feat /= col_max

        X = sp.hstack([X_tf, sp.csr_matrix(G_feat)], format="csr")
        return X, ids, labs

    def transform_text(
        self,
        title: str,
        abstract: str = "",
        graph_vec: Optional[np.ndarray] = None,
    ) -> sp.csr_matrix:
        """Transform a new paper (not in graph) for inference."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        combined = (f"{title} " * self.title_boost + abstract).strip()
        X_tf = self.vectorizer.transform([combined or "empty"])
        if not self.use_graph or graph_vec is None:
            return X_tf
        gv = sp.csr_matrix(graph_vec.reshape(1, -1).astype(np.float32))
        return sp.hstack([X_tf, gv], format="csr")

    def feature_names(self) -> List[str]:
        """All feature names: TF-IDF vocab + graph columns."""
        tf_names = list(self.vectorizer.get_feature_names_out())
        g_names  = [
            "in_deg_norm", "out_deg_norm", "tpr_norm", "auth_norm",
        ] + [f"dom_{d}" for d in DOMAIN_LIST]
        return tf_names + g_names

    @property
    def n_features(self) -> int:
        return self.vectorizer.vocabulary_.__len__() + GRAPH_DIM if self._fitted else 0

    @property
    def n_samples(self) -> int:
        return len(self.node_ids)
