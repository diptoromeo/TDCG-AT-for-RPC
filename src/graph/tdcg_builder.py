"""
src/graph/tdcg_builder.py
==========================
Temporal Directed Citation Graph builder.

Supports all 7 datasets via unified loading:
  - json_citing : our 3 datasets (arxiv/dblp/elsevier)
  - jsonl_ref   : DBLP-v10, S2ORC, PMC-OA  (JSONL, one paper per line)
  - json_ref    : ACL Anthology             (single JSON list)

Graph definition  G = (V, E, τ, w)
  V  : paper nodes
  E  : directed edges  (p_i → p_j)  iff p_i cites p_j  AND year(p_i) > year(p_j)
  τ  : E → ℤ    citing year
  w  : E → ℝ⁺   exp(−λ·Δt)   Eq.(3)
"""

from __future__ import annotations
import json
import math
import re
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

import networkx as nx
import numpy as np

from ..utils.taxonomy import (
    DOMAIN_LIST, N_DOMAINS,
    keyword_domain_score, assign_domain, DATASET_SCHEMAS,
)

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
YEAR_RE     = re.compile(r",\s*(20[012][0-9])\b")
YEAR_RANGE  = (2000, 2025)
LAMBDA      = 0.10
ALPHA_PR    = 0.85
MAX_PR_ITER = 200
BATCH_SIZE  = 50_000   # for large JSONL files


# ── helpers ───────────────────────────────────────────────────────────────────
def _nid(title: str, source: str) -> str:
    key = f"{source}::{title.strip().lower()[:120]}"
    return hashlib.md5(key.encode()).hexdigest()[:14]

def _clean(v) -> str:
    s = str(v or "").strip()
    return "" if s in ("nan", "None", "N/A", "") else s

def _year(authors_str: str) -> Optional[int]:
    for m in YEAR_RE.finditer(str(authors_str or "")):
        y = int(m.group(1))
        if YEAR_RANGE[0] <= y <= YEAR_RANGE[1]:
            return y
    return None

def _decay(yr_citing: int, yr_cited: int, lam: float = LAMBDA) -> float:
    """Eq.(3): w(e) = exp(−λ · Δt)"""
    return math.exp(-lam * max(0, yr_citing - yr_cited))

def _jsonl_stream(path: str, max_papers: int) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_papers:
                break
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ── main builder ──────────────────────────────────────────────────────────────
class TDCGBuilder:
    """
    Builds and stores G = (V, E, τ, w).

    Usage
    -----
    builder = TDCGBuilder()
    builder.load_json_citing("data/raw/arXiv_citation_dataset.json", "arxiv")
    builder.load_json_citing("data/raw/DBLP_citation_dataset.json",  "dblp")
    builder.load_jsonl_ref("data/external/dblp_v10.jsonl",           "dblp_v10")
    builder.load_json_ref("data/external/acl_anthology.json",        "acl")
    builder.finalize()
    builder.save("data/processed/tdcg.pkl")
    """

    def __init__(self, lambda_decay: float = LAMBDA,
                 alpha_pr: float = ALPHA_PR):
        self.G            = nx.DiGraph()
        self._title_map:  Dict[str, str] = {}   # title_key → nid
        self.lambda_decay = lambda_decay
        self.alpha_pr     = alpha_pr
        self._stats: dict = {}

    # ── loaders ───────────────────────────────────────────────────────────────
    def load_json_citing(self, path: str, source: str) -> "TDCGBuilder":
        """
        Load our 4 datasets.
        Each item has a root paper + list of citing_articles.
        Edge direction: citing_article → root  (newer cites older)
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]

        schema = DATASET_SCHEMAS.get(source, {})
        t_key  = schema.get("title",    "original_csv_title")
        a_key  = schema.get("abstract", "original_csv_abstract")
        at_key = schema.get("alt_title",    "scraped_main_title_from_scholar")
        aa_key = schema.get("alt_abstract", "scraped_main_abstract_from_scholar")
        c_key  = schema.get("citing_key",   "citing_articles")

        added_nodes = 0
        added_edges = 0
        for item in data:
            root_t   = _clean(item.get(t_key) or item.get(at_key, ""))
            root_a   = _clean(item.get(a_key) or item.get(aa_key, ""))
            root_auth= _clean(item.get("scraped_main_authors", ""))

            # infer root year from citers (root predates all citers)
            citer_yrs = [
                _year(ca.get("authors", ""))
                for ca in item.get(c_key, [])
                if _year(ca.get("authors", ""))
            ]
            root_yr = (min(citer_yrs) - 1) if citer_yrs else None

            root_nid = self._add_node(
                root_t or f"[unnamed:{source}]",
                root_a, root_auth, root_yr, source, is_root=True,
            )
            added_nodes += 1

            for ca in item.get(c_key, []):
                ca_t    = _clean(ca.get("title",    ""))
                ca_a    = _clean(ca.get("abstract", ""))
                ca_auth = _clean(ca.get("authors",  ""))
                ca_yr   = _year(ca_auth)
                if not ca_t:
                    continue
                ca_nid = self._add_node(ca_t, ca_a, ca_auth, ca_yr, source)
                added_nodes += 1

                if not self.G.has_edge(ca_nid, root_nid):
                    w = _decay(ca_yr, root_yr) if (ca_yr and root_yr) else 1.0
                    self.G.add_edge(
                        ca_nid, root_nid,
                        year=ca_yr,
                        delta_t=(ca_yr - root_yr) if (ca_yr and root_yr) else None,
                        weight=w,
                        source=source,
                    )
                    added_edges += 1

        logger.info(f"[{source}] json_citing: +{added_nodes} nodes, +{added_edges} edges "
                    f"(total {self.G.number_of_nodes()}/{self.G.number_of_edges()})")
        return self

    def load_jsonl_ref(self, path: str, source: str,
                       max_papers: int = 500_000) -> "TDCGBuilder":
        """
        Load DBLP-v10, S2ORC, PMC-OA.
        Each line: one paper with a references list.
        Edge direction: paper → reference  (paper cites reference)
        """
        schema   = DATASET_SCHEMAS.get(source, DATASET_SCHEMAS["dblp_v10"])
        t_key    = schema["title"]
        a_key    = schema["abstract"]
        yr_key   = schema.get("year",  "year")
        refs_key = schema.get("refs",  "references")

        # First pass: build year lookup
        year_lookup: Dict[str, int] = {}
        for item in _jsonl_stream(path, max_papers):
            t = _clean(item.get(t_key, ""))[:80]
            y = item.get(yr_key)
            if t and y:
                try:
                    year_lookup[t.lower()] = int(y)
                except (ValueError, TypeError):
                    pass

        added = 0
        for item in _jsonl_stream(path, max_papers):
            title    = _clean(item.get(t_key,  ""))
            abstract = _clean(item.get(a_key,  ""))
            year_raw = item.get(yr_key)
            try:
                year = int(year_raw) if year_raw else None
            except (ValueError, TypeError):
                year = None

            if not title:
                continue

            nid = self._add_node(title, abstract, "", year, source, is_root=True)
            added += 1

            for ref in item.get(refs_key, []):
                if isinstance(ref, dict):
                    ref_t = _clean(ref.get(t_key, ""))
                    try:
                        ref_y = int(ref.get(yr_key)) if ref.get(yr_key) else \
                                year_lookup.get(ref_t.lower()[:80])
                    except (ValueError, TypeError):
                        ref_y = year_lookup.get(ref_t.lower()[:80])
                elif isinstance(ref, str):
                    ref_t = ref
                    ref_y = year_lookup.get(ref_t.lower()[:80])
                else:
                    continue

                if not ref_t:
                    continue

                ref_nid = self._add_node(ref_t, "", "", ref_y, source)
                if not self.G.has_edge(nid, ref_nid):
                    valid = year and ref_y and year > ref_y
                    w     = _decay(year, ref_y) if valid else 1.0
                    self.G.add_edge(
                        nid, ref_nid,
                        year=year,
                        delta_t=(year - ref_y) if valid else None,
                        weight=w,
                        source=source,
                    )
            if added % 10_000 == 0:
                logger.info(f"[{source}] processed {added} papers …")

        logger.info(f"[{source}] jsonl_ref: {added} papers "
                    f"(total {self.G.number_of_nodes()}/{self.G.number_of_edges()})")
        return self

    def load_json_ref(self, path: str, source: str) -> "TDCGBuilder":
        """
        Load ACL Anthology (single JSON list).
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Handle {papers: [...]} wrapper
            data = data.get("papers", list(data.values())[0]
                            if isinstance(list(data.values())[0], list) else [data])

        schema   = DATASET_SCHEMAS.get(source, DATASET_SCHEMAS["acl"])
        t_key    = schema["title"]
        a_key    = schema["abstract"]
        yr_key   = schema.get("year",  "year")
        refs_key = schema.get("refs",  "outCitations")

        for item in data:
            title    = _clean(item.get(t_key,  ""))
            abstract = _clean(item.get(a_key,  ""))
            try:
                year = int(item.get(yr_key)) if item.get(yr_key) else None
            except (ValueError, TypeError):
                year = None

            if not title:
                continue
            nid = self._add_node(title, abstract, "", year, source, is_root=True)
            for ref_id in item.get(refs_key, []):
                if isinstance(ref_id, str):
                    ref_nid = self._add_node(ref_id, "", "", None, source)
                    if not self.G.has_edge(nid, ref_nid):
                        self.G.add_edge(nid, ref_nid, year=year,
                                        weight=1.0, source=source)

        logger.info(f"[{source}] json_ref: {len(data)} papers "
                    f"(total {self.G.number_of_nodes()}/{self.G.number_of_edges()})")
        return self

    # ── node creation ─────────────────────────────────────────────────────────
    def _add_node(self, title: str, abstract: str = "",
                  authors: str = "", year: Optional[int] = None,
                  source: str = "", is_root: bool = False) -> str:
        key = title.strip().lower()[:120]
        if key in self._title_map:
            return self._title_map[key]

        nid = _nid(title, source)
        # collision resolution
        while nid in self.G:
            nid += "x"

        dom_scores = keyword_domain_score(title, abstract)
        best_dom   = max(dom_scores, key=dom_scores.get)
        domain     = best_dom if dom_scores[best_dom] >= 0.01 else "Unknown"

        self.G.add_node(nid,
            title    = title[:300],
            abstract = abstract[:500],
            authors  = authors[:200],
            year     = year,
            source   = source,
            domain   = domain,
            is_root  = is_root,
            **{f"dom_{d}": round(dom_scores[d], 5) for d in DOMAIN_LIST},
        )
        self._title_map[key] = nid
        return nid

    # ── finalize ──────────────────────────────────────────────────────────────
    def finalize(self) -> "TDCGBuilder":
        """Compute T-PageRank, HITS, degree features, WCC clusters."""
        logger.info("Computing T-PageRank …")
        try:
            pr = nx.pagerank(self.G, alpha=self.alpha_pr,
                             weight="weight", max_iter=MAX_PR_ITER)
        except Exception:
            pr = {n: 1.0 / max(self.G.number_of_nodes(), 1) for n in self.G}

        max_pr = max(pr.values()) if pr else 1.0
        for n, v in pr.items():
            self.G.nodes[n]["tpr"]      = round(v, 8)
            self.G.nodes[n]["tpr_norm"] = round(v / max_pr, 6)

        logger.info("Computing HITS …")
        try:
            hubs, auth = nx.hits(self.G, max_iter=200, normalized=True)
        except Exception:
            hubs = auth = {n: 0.0 for n in self.G}

        max_h = max(hubs.values()) if hubs else 1.0
        max_a = max(auth.values()) if auth else 1.0
        for n in self.G:
            self.G.nodes[n]["hub"]       = round(hubs.get(n, 0), 8)
            self.G.nodes[n]["authority"] = round(auth.get(n, 0), 8)
            self.G.nodes[n]["hub_norm"]  = round(hubs.get(n, 0) / max_h, 6)
            self.G.nodes[n]["auth_norm"] = round(auth.get(n, 0) / max_a, 6)

        logger.info("Computing degree features …")
        max_in  = max((self.G.in_degree(n)  for n in self.G), default=1)
        max_out = max((self.G.out_degree(n) for n in self.G), default=1)
        for n in self.G:
            ind  = self.G.in_degree(n)
            outd = self.G.out_degree(n)
            self.G.nodes[n]["in_degree"]      = ind
            self.G.nodes[n]["out_degree"]     = outd
            self.G.nodes[n]["in_deg_norm"]    = ind  / (max_in  + 1e-12)
            self.G.nodes[n]["out_deg_norm"]   = outd / (max_out + 1e-12)

        logger.info("Computing WCC clusters …")
        undirected = self.G.to_undirected()
        components = list(nx.connected_components(undirected))
        max_sz     = max((len(c) for c in components), default=1)
        for cid, comp in enumerate(components):
            for n in comp:
                self.G.nodes[n]["cluster_id"]        = cid
                self.G.nodes[n]["cluster_size"]      = len(comp)
                self.G.nodes[n]["cluster_size_norm"] = len(comp) / max_sz

        # Temporal statistics
        years  = [d["year"] for _, d in self.G.nodes(data=True) if d.get("year")]
        deltas = [d["delta_t"] for _, _, d in self.G.edges(data=True)
                  if d.get("delta_t") is not None and d["delta_t"] >= 0]

        self._stats = {
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "n_wcc":   len(components),
            "largest_wcc": max((len(c) for c in components), default=0),
            "year_range": (min(years), max(years)) if years else (None, None),
            "n_with_year": len(years),
            "delta_t_mean": round(float(np.mean(deltas)), 3) if deltas else None,
            "delta_t_std":  round(float(np.std(deltas)),  3) if deltas else None,
            "lambda_decay": self.lambda_decay,
            "alpha_pr":     self.alpha_pr,
        }
        logger.info(f"Graph finalized: {self._stats}")
        return self

    @property
    def stats(self) -> dict:
        return self._stats

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"G": self.G, "title_map": self._title_map,
                         "stats": self._stats}, f, protocol=4)
        logger.info(f"Graph saved → {path}")

    @classmethod
    def load(cls, path: str) -> "TDCGBuilder":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj.G            = d["G"]
        obj._title_map   = d["title_map"]
        obj._stats       = d.get("stats", {})
        obj.lambda_decay = obj._stats.get("lambda_decay", LAMBDA)
        obj.alpha_pr     = obj._stats.get("alpha_pr", ALPHA_PR)
        return obj
