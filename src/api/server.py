"""
src/api/server.py
==================
FastAPI REST API for the ClassifierAgent system.

Endpoints
---------
GET  /                        system health + stats
GET  /domains                 list all 9 domain classes
POST /classify                classify single paper
POST /classify/batch          classify multiple papers
GET  /classify/domain/{d}     list papers in domain d
GET  /stats                   graph + model statistics
POST /evaluate                run evaluation suite
GET  /eval/results            return last evaluation results
"""

from __future__ import annotations
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PaperIn(BaseModel):
    title:    str  = Field(..., min_length=3, description="Paper title")
    abstract: str  = Field("",                description="Paper abstract")
    year:     Optional[int]   = None
    source:   Optional[str]   = None

class BatchIn(BaseModel):
    papers: List[PaperIn] = Field(..., max_items=500)

class ClassifyOut(BaseModel):
    domain:               str
    confidence:           float
    all_scores:           Dict[str, float]
    keyword_domain:       str
    is_interdisciplinary: bool
    elapsed_ms:           float

class EvalRequest(BaseModel):
    protocol: str = Field("all",
        description="temporal_split | cross_validation | loso | ablation | all")

# ── global system state ───────────────────────────────────────────────────────
_SYS: Dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or build the full pipeline on startup."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from src.graph.tdcg_builder      import TDCGBuilder
    from src.graph.feature_extractor import TDCGFeatureExtractor
    from src.agent.classifier_agent  import ClassifierAgent
    from src.evaluation.evaluator    import Evaluator

    data_dir   = os.getenv("DATA_DIR",    "data/raw")
    proc_dir   = os.getenv("PROC_DIR",    "data/processed")
    ext_dir    = os.getenv("EXT_DIR",     "data/external")
    graph_pkl  = os.path.join(proc_dir, "tdcg.pkl")
    agent_pkl  = os.path.join(proc_dir, "agent.pkl")
    extractor_pkl = os.path.join(proc_dir, "extractor.pkl")

    Path(proc_dir).mkdir(parents=True, exist_ok=True)

    # ── load or build graph ──────────────────────────────────────────────────
    if Path(graph_pkl).exists():
        logger.info("Loading cached graph …")
        builder = TDCGBuilder.load(graph_pkl)
    else:
        logger.info("Building graph from scratch …")
        builder = TDCGBuilder()
        for fname, schema in [
            ("arXiv_citation_dataset.json",    "arxiv"),
            ("DBLP_citation_dataset.json",     "dblp"),
            ("Elsevier_citation_dataset.json", "elsevier"),
        ]:
            p = os.path.join(data_dir, fname)
            if Path(p).exists():
                builder.load_json_citing(p, schema)

        # external datasets (loaded if present)
        ext_map = {
            "dblp_v10.jsonl":         ("dblp_v10",  "jsonl_ref"),
            "acl_anthology.json":     ("acl",        "json_ref"),
            "s2orc_subset.jsonl":     ("s2orc",      "jsonl_ref"),
            "pmc_oa.jsonl":           ("pmc",        "jsonl_ref"),
        }
        for fname, (schema, fmt) in ext_map.items():
            p = os.path.join(ext_dir, fname)
            if Path(p).exists():
                logger.info(f"Loading external: {fname}")
                if fmt == "jsonl_ref":
                    max_p = int(os.getenv("MAX_EXT_PAPERS", "200000"))
                    builder.load_jsonl_ref(p, schema, max_papers=max_p)
                else:
                    builder.load_json_ref(p, schema)

        builder.finalize()
        builder.save(graph_pkl)

    # ── load or fit extractor + agent ────────────────────────────────────────
    if Path(agent_pkl).exists() and Path(extractor_pkl).exists():
        logger.info("Loading cached extractor + agent …")
        import pickle
        with open(extractor_pkl, "rb") as f:
            extractor = pickle.load(f)
        agent = ClassifierAgent.load(agent_pkl)
    else:
        logger.info("Fitting extractor + agent …")
        extractor = TDCGFeatureExtractor()
        extractor.fit(builder)
        X, node_ids, labels = extractor.transform(builder)

        agent = ClassifierAgent()
        agent.fit(X, labels)

        import pickle
        with open(extractor_pkl, "wb") as f:
            pickle.dump(extractor, f, protocol=4)
        agent.save(agent_pkl)

    # ── build corpus list for evaluator ─────────────────────────────────────
    corpus = []
    G = builder.G
    for nid, lab, src, yr in zip(
        extractor.node_ids, extractor.labels,
        extractor.sources, extractor.years,
    ):
        corpus.append({
            "nid": nid, "label": lab, "source": src, "year": yr,
            "split": "test" if (yr and yr >= 2023) else "train",
        })

    evaluator = Evaluator(agent, extractor, builder, corpus)

    _SYS.update({
        "builder":   builder,
        "extractor": extractor,
        "agent":     agent,
        "evaluator": evaluator,
        "corpus":    corpus,
        "eval_results": None,
    })
    logger.info("✅ ClassifierAgent system ready.")
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="TDCG-AT ClassifierAgent API",
    description=(
        "Agentic AI paper classification system based on "
        "Temporal Directed Citation Graph (TDCG-AT). "
        "9-class domain taxonomy · 3 evaluation protocols."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def health():
    b = _SYS.get("builder")
    st = b.stats if b else {}
    return {
        "status":  "running",
        "version": "2.0.0",
        "graph":   st,
        "agent": {
            "classes":   _SYS["agent"].classes_  if _SYS.get("agent") else [],
            "n_samples": _SYS["extractor"].n_samples if _SYS.get("extractor") else 0,
            "n_features":_SYS["extractor"].n_features if _SYS.get("extractor") else 0,
        } if _SYS.get("agent") else {},
    }


@app.get("/domains", tags=["Classify"])
async def list_domains():
    from src.utils.taxonomy import DOMAIN_LIST, DOMAIN_DESC
    return {
        "domains": [
            {"id": d, "description": DOMAIN_DESC.get(d, d)}
            for d in DOMAIN_LIST
        ],
        "n_domains": len(DOMAIN_LIST),
    }


@app.post("/classify", response_model=ClassifyOut, tags=["Classify"])
async def classify_paper(paper: PaperIn):
    t0 = time.perf_counter()
    ag = _SYS.get("agent")
    ex = _SYS.get("extractor")
    if not ag or not ex:
        raise HTTPException(503, "System not ready")
    try:
        result = ag.classify_paper(
            title=paper.title, abstract=paper.abstract,
            extractor=ex, builder=_SYS["builder"],
        )
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/classify/batch", tags=["Classify"])
async def classify_batch(req: BatchIn):
    t0  = time.perf_counter()
    ag  = _SYS.get("agent")
    ex  = _SYS.get("extractor")
    if not ag or not ex:
        raise HTTPException(503, "System not ready")

    results = []
    for paper in req.papers:
        try:
            r = ag.classify_paper(paper.title, paper.abstract, ex,
                                   _SYS["builder"])
            results.append(r)
        except Exception as e:
            results.append({"error": str(e)})

    return {
        "results":    results,
        "total":      len(results),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/classify/domain/{domain}", tags=["Classify"])
async def papers_in_domain(
    domain: str,
    top_k:  int = Query(20, ge=1, le=200),
):
    G  = _SYS["builder"].G
    papers = [
        {
            "title":     d.get("title", ""),
            "year":      d.get("year"),
            "source":    d.get("source", ""),
            "tpr":       d.get("tpr", 0.0),
            "in_degree": d.get("in_degree", 0),
        }
        for n, d in G.nodes(data=True)
        if d.get("domain") == domain
    ]
    papers.sort(key=lambda x: x["tpr"], reverse=True)
    return {"domain": domain, "total": len(papers), "papers": papers[:top_k]}


@app.get("/stats", tags=["System"])
async def stats():
    return {
        "graph":      _SYS["builder"].stats if _SYS.get("builder") else {},
        "n_classes":  len(_SYS["agent"].classes_)      if _SYS.get("agent") else 0,
        "classes":    _SYS["agent"].classes_            if _SYS.get("agent") else [],
        "n_samples":  _SYS["extractor"].n_samples       if _SYS.get("extractor") else 0,
        "n_features": _SYS["extractor"].n_features      if _SYS.get("extractor") else 0,
        "eval_ready": _SYS.get("eval_results") is not None,
    }


@app.post("/evaluate", tags=["Evaluate"])
async def run_evaluation(
    req: EvalRequest,
    background_tasks: BackgroundTasks,
):
    ev = _SYS.get("evaluator")
    if not ev:
        raise HTTPException(503, "Evaluator not ready")

    def _run():
        proto = req.protocol
        if proto == "all":
            results = ev.run_all()
        elif proto == "temporal_split":
            results = {"temporal_split": ev.temporal_split()}
        elif proto == "cross_validation":
            results = {"cross_validation": ev.cross_validation()}
        elif proto == "loso":
            results = {"leave_one_source_out": ev.leave_one_source_out()}
        elif proto == "ablation":
            results = {"ablation_study": ev.ablation_study()}
        else:
            return
        _SYS["eval_results"] = results

    background_tasks.add_task(_run)
    return {"status": "evaluation started", "protocol": req.protocol,
            "message": "Call GET /eval/results to retrieve once complete."}


@app.get("/eval/results", tags=["Evaluate"])
async def get_eval_results():
    r = _SYS.get("eval_results")
    if r is None:
        raise HTTPException(404, "No evaluation results yet. POST /evaluate first.")
    return r


@app.exception_handler(Exception)
async def global_error(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500,
                        content={"error": str(exc), "type": type(exc).__name__})
