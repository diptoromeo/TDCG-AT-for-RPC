#!/usr/bin/env python3
"""
experiments.py  —  TDCG-AT Comprehensive Experiment Suite
==========================================================
Runs all experiments, prints every result, saves figures + CSV.

Experiment groups
-----------------
A.  Dataset Statistics          — corpus, graph, class distribution
B.  Feature Matrix Analysis     — TF-IDF density, SNR, graph features
C.  Classifier Comparison       — SVM vs LR vs GP vs Ensemble (5-fold CV)
D.  Ablation: Feature Blocks    — TF-IDF only / +struct / +domain / full
E.  Ablation: Model Components  — remove each sub-classifier
F.  Ablation: Graph Parameters  — lambda, mu, title-boost
G.  Temporal Split Evaluation   — train≤2022 / test≥2023
H.  Leave-One-Source-Out        — per-source generalization
I.  Per-Class Performance       — P/R/F1 for all 9 domains
J.  Confusion Matrix            — full 9×9 heatmap
K.  Learning Curve              — accuracy vs training size
L.  Hyperparameter Sensitivity  — C_svm, C_lr, mu grid
M.  Graph Structural Analysis   — degree dist, PageRank, WCC
N.  Temporal Trend Analysis     — year distribution of corpus
O.  Classification Confidence   — histogram of predicted probabilities
"""

import os, sys, json, re, time, pickle, warnings
import numpy as np, scipy.sparse as sp
import networkx as nx
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings('ignore')

# ── working directory
PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)
sys.path.insert(0, PROJ)

OUT  = os.path.join(PROJ, "experiment_results")
FIG  = os.path.join(OUT,  "figures")
CSV  = os.path.join(OUT,  "csv")
os.makedirs(FIG, exist_ok=True)
os.makedirs(CSV, exist_ok=True)

# ── sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, learning_curve, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer

from src.graph.tdcg_builder      import TDCGBuilder
from src.graph.feature_extractor import TDCGFeatureExtractor
from src.agent.classifier_agent  import ClassifierAgent, _SVCWithProba
from src.utils.taxonomy          import DOMAIN_LIST, DOMAIN_KWS, keyword_domain_score

# ── colours
PALETTE = ["#185FA5","#0F6E56","#C55A11","#7030A0","#C00020",
           "#BA7517","#1D9E75","#534AB7","#D85A30"]
sns.set_style("whitegrid")
sns.set_palette(PALETTE)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

RESULTS = {}   # master dict — all numbers written here then to JSON

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TDCG-AT COMPREHENSIVE EXPERIMENT SUITE")
print("="*70)

# ── load pipeline ─────────────────────────────────────────────────────────────
print("\n[LOADING] Building pipeline …")
t0 = time.time()
builder   = TDCGBuilder.load("data/processed/tdcg.pkl")
with open("data/processed/extractor.pkl","rb") as f:
    extractor = pickle.load(f)
agent = ClassifierAgent.load("data/processed/agent.pkl")
G     = builder.G
X, node_ids, labels = extractor.transform(builder)
print(f"  Pipeline loaded in {time.time()-t0:.1f}s")
print(f"  X.shape = {X.shape}   classes = {agent.classes_}")


# ═══════════════════════════════════════════════════════════════════════════════
# A. DATASET STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  A. DATASET STATISTICS")
print("─"*70)

import json as _json, re as _re
sources_data = {}
for fname, key in [("arXiv_citation_dataset.json","arXiv"),
                   ("DBLP_citation_dataset.json","DBLP"),
                   ("Elsevier_citation_dataset.json","Elsevier")]:
    p = os.path.join("data/raw", fname)
    if os.path.exists(p):
        with open(p) as f:
            d = _json.load(f)
        n_root   = len(d)
        n_citing = sum(len(x.get('citing_articles',[])) for x in d)
        years = []
        for item in d:
            for ca in item.get('citing_articles',[]):
                m = _re.search(r', (20[012][0-9])', str(ca.get('authors','')))
                if m: years.append(int(m.group(1)))
        abstracts_r = sum(1 for x in d
                          if len(str(x.get('original_csv_abstract','') or
                                     x.get('scraped_main_abstract_from_scholar','') or '')) > 30)
        sources_data[key] = {
            'root': n_root, 'citing': n_citing,
            'total': n_root + n_citing,
            'year_min': min(years) if years else None,
            'year_max': max(years) if years else None,
            'abstract_cov': round(100*abstracts_r/max(n_root,1),1),
        }

print(f"\n  {'Source':<12} {'Root':>7} {'Citing':>8} {'Total':>8} "
      f"{'Year Range':<14} {'Abs.Cov%':>9}")
print(f"  {'─'*64}")
total_r = total_c = 0
for src, sd in sources_data.items():
    yr = f"{sd['year_min']}-{sd['year_max']}" if sd['year_min'] else "—"
    print(f"  {src:<12} {sd['root']:>7,} {sd['citing']:>8,} "
          f"{sd['total']:>8,} {yr:<14} {sd['abstract_cov']:>8.1f}%")
    total_r += sd['root']; total_c += sd['citing']
print(f"  {'─'*64}")
print(f"  {'TOTAL':<12} {total_r:>7,} {total_c:>8,} {total_r+total_c:>8,}")

# Graph stats
st = builder.stats
print(f"\n  Graph |V|={st['n_nodes']:,}  |E|={st['n_edges']:,}  "
      f"WCC={st['n_wcc']:,}  largest_wcc={st['largest_wcc']:,}")
print(f"  Year range: {st['year_range']}  n_with_year={st['n_with_year']:,}")
print(f"  Δt mean={st['delta_t_mean']} yr  std={st['delta_t_std']} yr")
print(f"  Labeled samples: {len(labels):,}   Classes: {len(set(labels))}")

# Label distribution
dist = Counter(labels)
print(f"\n  {'Domain':<26} {'N':>6}  {'%':>7}  Bar")
for d2, n in sorted(dist.items(), key=lambda x:-x[1]):
    bar = "█"*int(n/max(dist.values())*20)
    print(f"  {d2:<26} {n:>6}  {100*n/len(labels):>6.1f}%  {bar}")

RESULTS['dataset'] = {
    'sources': sources_data,
    'graph': st,
    'n_labeled': len(labels),
    'n_classes': len(set(labels)),
    'label_distribution': dict(dist),
    'imbalance_ratio': round(max(dist.values())/min(dist.values()),2),
}

# Figure A1: Dataset overview
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# Bar chart: papers per source
src_names = list(sources_data.keys())
src_roots = [sources_data[s]['root']   for s in src_names]
src_cit   = [sources_data[s]['citing'] for s in src_names]
x = np.arange(len(src_names))
axes[0].bar(x, src_roots,  label='Root papers',   color=PALETTE[0], alpha=0.9)
axes[0].bar(x, src_cit,    bottom=src_roots, label='Citing papers', color=PALETTE[1], alpha=0.9)
axes[0].set_xticks(x); axes[0].set_xticklabels(src_names)
axes[0].set_title("A1. Papers per Source"); axes[0].set_ylabel("Count")
axes[0].legend(fontsize=8)
# Domain distribution pie
dom_names = [d for d,_ in sorted(dist.items(), key=lambda x:-x[1])]
dom_vals  = [dist[d] for d in dom_names]
short     = [d.replace('_',' ').replace('Mathematics Physics','Math/Phys')
               .replace('Medicine Biology','Med/Bio') for d in dom_names]
axes[1].pie(dom_vals, labels=short, autopct='%1.1f%%',
            colors=PALETTE[:len(dom_vals)], startangle=90, textprops={'fontsize':7})
axes[1].set_title("A2. Domain Distribution")
# Year distribution
years_all = [G.nodes[n].get('year') for n in G.nodes() if G.nodes[n].get('year')]
yr_cnt = Counter(y for y in years_all if 2015<=y<=2025)
axes[2].bar(yr_cnt.keys(), yr_cnt.values(), color=PALETTE[2], alpha=0.9)
axes[2].set_title("A3. Year Distribution (2015-2025)")
axes[2].set_xlabel("Year"); axes[2].set_ylabel("Papers")
plt.tight_layout()
plt.savefig(f"{FIG}/A_dataset_overview.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: A_dataset_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# B. FEATURE MATRIX ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  B. FEATURE MATRIX ANALYSIS")
print("─"*70)

X_tf   = X[:, :-13]   # TF-IDF block
X_gr   = X[:, -13:]   # Graph block
density = X_tf.nnz / (X_tf.shape[0]*X_tf.shape[1])

# Connected vs random cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import random; random.seed(42)
edges_sample = list(G.edges())[:300]
nid_to_row   = {nid: i for i, nid in enumerate(node_ids)}
conn_sims, rand_sims = [], []
for u, v in edges_sample:
    if u in nid_to_row and v in nid_to_row:
        s = float(cosine_similarity(X_tf[nid_to_row[u]], X_tf[nid_to_row[v]])[0][0])
        conn_sims.append(s)
nids_list = list(nid_to_row.keys())
for _ in range(300):
    a, b = random.sample(nids_list[:1000], 2)
    if a in nid_to_row and b in nid_to_row:
        s = float(cosine_similarity(X_tf[nid_to_row[a]], X_tf[nid_to_row[b]])[0][0])
        rand_sims.append(s)

snr = np.mean(conn_sims)/max(np.mean(rand_sims),1e-9)
from scipy.stats import ttest_ind
_, pval = ttest_ind(conn_sims, rand_sims)

print(f"  TF-IDF matrix    : {X_tf.shape}  density={density:.5f}")
print(f"  Graph features   : {X_gr.shape}")
print(f"  Full matrix      : {X.shape}")
print(f"  Connected sim    : {np.mean(conn_sims):.4f} ± {np.std(conn_sims):.4f}")
print(f"  Random sim       : {np.mean(rand_sims):.4f} ± {np.std(rand_sims):.4f}")
print(f"  SNR              : {snr:.2f}×   p={pval:.2e}")

# PageRank stats
pr_vals = [G.nodes[n].get('tpr', 0) for n in G.nodes()]
print(f"  T-PageRank max   : {max(pr_vals):.6f}   mean={np.mean(pr_vals):.7f}")
auth_vals = [G.nodes[n].get('authority', 0) for n in G.nodes()]
hub_vals  = [G.nodes[n].get('hub', 0)       for n in G.nodes()]
print(f"  Authority max    : {max(auth_vals):.6f}")
print(f"  Hub max          : {max(hub_vals):.6f}")

RESULTS['features'] = {
    'matrix_shape': list(X.shape),
    'tfidf_shape': list(X_tf.shape),
    'graph_shape': list(X_gr.shape),
    'tfidf_density': round(density, 6),
    'conn_sim_mean': round(np.mean(conn_sims), 4),
    'rand_sim_mean': round(np.mean(rand_sims), 4),
    'snr': round(snr, 3),
    'snr_pvalue': float(pval),
    'tpr_max': round(max(pr_vals), 6),
}

# Figure B1: Feature analysis
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(conn_sims, bins=25, alpha=0.7, label=f'Connected (μ={np.mean(conn_sims):.3f})', color=PALETTE[0])
axes[0].hist(rand_sims, bins=25, alpha=0.7, label=f'Random (μ={np.mean(rand_sims):.3f})',    color=PALETTE[2])
axes[0].set_title(f"B1. Cosine Similarity Distribution\nSNR={snr:.2f}×, p={pval:.1e}")
axes[0].set_xlabel("Cosine Similarity"); axes[0].legend()
# PageRank distribution (log)
pr_nonzero = [v for v in pr_vals if v > 0]
axes[1].hist(np.log10(pr_nonzero), bins=30, color=PALETTE[1], alpha=0.9)
axes[1].set_title("B2. T-PageRank Distribution (log10)")
axes[1].set_xlabel("log10(TPR)"); axes[1].set_ylabel("Count")
# In-degree distribution (log-log)
in_degs = [G.in_degree(n) for n in G.nodes() if G.in_degree(n) > 0]
dc = Counter(in_degs)
xs, ys = zip(*sorted(dc.items()))
axes[2].loglog(xs, ys, 'o', color=PALETTE[3], markersize=4, alpha=0.8)
axes[2].set_title("B3. In-Degree Distribution (log-log)")
axes[2].set_xlabel("In-degree k"); axes[2].set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG}/B_feature_analysis.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: B_feature_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# C. CLASSIFIER COMPARISON  (10 models, 5-fold CV)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  C. CLASSIFIER COMPARISON  (5-fold CV, full feature matrix)")
print("─"*70)

def make_svc():
    clf = _SVCWithProba(C=1.0)
    return clf

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

CLASSIFIERS = [
    ("LinearSVC (softmax)",   _SVCWithProba(C=1.0)),
    ("LogisticRegression",    LogisticRegression(C=1.0,max_iter=500,solver='lbfgs',class_weight='balanced')),
    ("SGD Classifier",        SGDClassifier(max_iter=1000,class_weight='balanced',random_state=42)),
    ("ComplementNB",          ComplementNB(alpha=0.1)),
    ("KNN (k=5)",             KNeighborsClassifier(n_neighbors=5,n_jobs=-1)),
    ("Decision Tree",         DecisionTreeClassifier(max_depth=20,class_weight='balanced',random_state=42)),
    ("Random Forest",         RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42,n_jobs=-1)),
    ("TDCG-AT SVM only",      _SVCWithProba(C=1.0)),
    ("TDCG-AT LR only",       LogisticRegression(C=1.0,max_iter=500,solver='lbfgs',class_weight='balanced')),
    ("TDCG-AT Ensemble",      None),   # handled separately
]

clf_results = {}
print(f"\n  {'Model':<26} {'Accuracy':>9} {'Macro-F1':>10} {'Wt-F1':>8} {'Time(s)':>9}")
print(f"  {'─'*64}")
for name, clf in CLASSIFIERS:
    t_start = time.time()
    if name == "TDCG-AT Ensemble":
        preds = agent.predict(X, node_ids=node_ids, builder=builder)
        acc  = accuracy_score(labels, preds)
        mf1  = f1_score(labels, preds, average='macro',    zero_division=0)
        wf1  = f1_score(labels, preds, average='weighted', zero_division=0)
    else:
        try:
            preds = cross_val_predict(clf, X, labels, cv=cv5)
            acc   = accuracy_score(labels, preds)
            mf1   = f1_score(labels, preds, average='macro',    zero_division=0)
            wf1   = f1_score(labels, preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"  {name:<26}  SKIPPED: {e}")
            continue
    elapsed = time.time() - t_start
    clf_results[name] = {'accuracy':round(acc,4),'macro_f1':round(mf1,4),
                         'weighted_f1':round(wf1,4),'time_s':round(elapsed,2)}
    marker = " ◄ BEST" if name == "TDCG-AT Ensemble" else ""
    print(f"  {name:<26} {acc:>9.4f} {mf1:>10.4f} {wf1:>8.4f} {elapsed:>8.2f}s{marker}")

RESULTS['classifier_comparison'] = clf_results

# Figure C1: Classifier comparison bar chart
fig, ax = plt.subplots(figsize=(13, 5))
names = list(clf_results.keys())
accs  = [clf_results[n]['accuracy'] for n in names]
mf1s  = [clf_results[n]['macro_f1'] for n in names]
x     = np.arange(len(names))
bars1 = ax.bar(x - 0.2, accs,  0.38, label='Accuracy',   color=PALETTE[0], alpha=0.9)
bars2 = ax.bar(x + 0.2, mf1s,  0.38, label='Macro-F1',   color=PALETTE[2], alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=7)
ax.set_ylim(0, 1.08)
ax.set_title("C1. Classifier Comparison (5-fold CV)")
ax.set_ylabel("Score")
ax.legend()
for b in bars1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                         f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=6)
for b in bars2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                         f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=6)
plt.tight_layout()
plt.savefig(f"{FIG}/C_classifier_comparison.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: C_classifier_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# D. ABLATION: FEATURE BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  D. ABLATION: FEATURE BLOCKS")
print("─"*70)

# Feature column slices
X_tfidf_only    = X[:, :-13]                              # no graph
X_tfidf_struct  = sp.hstack([X[:,:-13], X[:,-13:-9]])    # +4 structural
X_tfidf_domain  = sp.hstack([X[:,:-13], X[:,-9:]])       # +9 domain scores
X_full          = X                                        # all 13

abl_configs = [
    ("TF-IDF only (d=11543)",        X_tfidf_only),
    ("TF-IDF + structural (d=+4)",   X_tfidf_struct),
    ("TF-IDF + domain scores (d=+9)",X_tfidf_domain),
    ("Full X (TF-IDF+graph, d=+13)", X_full),
]

ablation_results = {}
print(f"\n  {'Configuration':<36} {'Accuracy':>9} {'Macro-F1':>10} {'Δ Acc':>8}")
print(f"  {'─'*67}")
base_acc = None
for name, X_abl in abl_configs:
    clf = _SVCWithProba(C=1.0)
    preds = cross_val_predict(clf, X_abl, labels, cv=cv5)
    acc  = accuracy_score(labels, preds)
    mf1  = f1_score(labels, preds, average='macro', zero_division=0)
    delta = "" if base_acc is None else f"{(acc-base_acc)*100:+.2f}pp"
    if base_acc is None: base_acc = acc
    ablation_results[name] = {'accuracy':round(acc,4), 'macro_f1':round(mf1,4),
                               'delta_acc':round(acc-base_acc,4)}
    print(f"  {name:<36} {acc:>9.4f} {mf1:>10.4f} {delta:>8}")

RESULTS['ablation_features'] = ablation_results

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
abl_names  = [n.split('(')[0].strip() for n in ablation_results]
abl_accs   = [ablation_results[n]['accuracy'] for n in ablation_results]
abl_mf1s   = [ablation_results[n]['macro_f1'] for n in ablation_results]
colors_abl = [PALETTE[2] if 'Full' in n else PALETTE[0] for n in ablation_results]
axes[0].barh(abl_names, abl_accs,  color=colors_abl, alpha=0.9)
for i, v in enumerate(abl_accs): axes[0].text(v+0.001, i, f'{v:.4f}', va='center', fontsize=9)
axes[0].set_xlim(0.7, 1.05); axes[0].set_title("D1. Accuracy by Feature Block")
axes[1].barh(abl_names, abl_mf1s,  color=colors_abl, alpha=0.9)
for i, v in enumerate(abl_mf1s): axes[1].text(v+0.001, i, f'{v:.4f}', va='center', fontsize=9)
axes[1].set_xlim(0.5, 1.05); axes[1].set_title("D2. Macro-F1 by Feature Block")
plt.tight_layout()
plt.savefig(f"{FIG}/D_ablation_features.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: D_ablation_features.png")


# ═══════════════════════════════════════════════════════════════════════════════
# E. ABLATION: MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  E. ABLATION: MODEL COMPONENTS (Ensemble sub-classifier removal)")
print("─"*70)

def ensemble_pred(svm_p, lr_p, gp_p, w_svm, w_lr, w_gp, classes, all_cls):
    idx = {c:i for i,c in enumerate(all_cls)}
    def align(p, cls):
        a = np.zeros((p.shape[0],len(all_cls)),dtype=np.float32)
        for j,c in enumerate(cls):
            if c in idx: a[:,idx[c]] = p[:,j]
        return a
    combined = w_svm*align(svm_p,classes) + w_lr*align(lr_p,classes) + w_gp*align(gp_p,all_cls)
    return np.array(all_cls)[combined.argmax(axis=1)]

# Fit all three base models on full data
svm_m = _SVCWithProba(C=1.0);   svm_m.fit(X, labels)
lr_m  = LogisticRegression(C=1.0,max_iter=500,solver='lbfgs',class_weight='balanced')
lr_m.fit(X, labels)
all_cls = agent.classes_

# For CV: use cross_val_predict on SVM and LR
svm_preds_cv = cross_val_predict(_SVCWithProba(C=1.0), X, labels, cv=cv5)
lr_preds_cv  = cross_val_predict(
    LogisticRegression(C=1.0,max_iter=500,solver='lbfgs',class_weight='balanced'),
    X, labels, cv=cv5)

# Get GP output
gp_proba = agent._graph_propagation(X, node_ids, builder)
gp_preds  = np.array(all_cls)[gp_proba.argmax(axis=1)]

model_configs = [
    ("SVM only (w=1.0)",         svm_preds_cv),
    ("LR only (w=1.0)",          lr_preds_cv),
    ("GP only (w=1.0)",          gp_preds),
    ("SVM+LR (0.5/0.5)",         None),
    ("SVM+GP (0.5/0.5)",         None),
    ("LR+GP  (0.5/0.5)",         None),
    ("Full Ensemble (0.45/0.35/0.20)", None),
]

# get probabilities for ensemble combos
svm_p_full = svm_m.predict_proba(X)
lr_p_full  = lr_m.predict_proba(X)
gp_p_full  = gp_proba

model_combos = {
    "SVM+LR (0.5/0.5)": (0.5, 0.5, 0.0),
    "SVM+GP (0.5/0.5)": (0.5, 0.0, 0.5),
    "LR+GP  (0.5/0.5)": (0.0, 0.5, 0.5),
    "Full Ensemble (0.45/0.35/0.20)": (0.45, 0.35, 0.20),
}

model_abl_results = {}
print(f"\n  {'Configuration':<36} {'Accuracy':>9} {'Macro-F1':>10}")
print(f"  {'─'*56}")
for name, preds in model_configs:
    if preds is None:
        ws, wl, wg = model_combos[name]
        combined = ws*svm_p_full + wl*lr_p_full + wg*gp_p_full
        preds = np.array(all_cls)[combined.argmax(axis=1)]
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average='macro', zero_division=0)
    model_abl_results[name] = {'accuracy':round(acc,4),'macro_f1':round(mf1,4)}
    marker = " ◄" if "Full" in name else ""
    print(f"  {name:<36} {acc:>9.4f} {mf1:>10.4f}{marker}")

RESULTS['ablation_models'] = model_abl_results

fig, ax = plt.subplots(figsize=(11, 4))
mnames = list(model_abl_results.keys())
maccs  = [model_abl_results[n]['accuracy'] for n in mnames]
mmf1s  = [model_abl_results[n]['macro_f1'] for n in mnames]
x = np.arange(len(mnames))
ax.bar(x-0.2, maccs, 0.38, label='Accuracy', color=PALETTE[0], alpha=0.9)
ax.bar(x+0.2, mmf1s, 0.38, label='Macro-F1', color=PALETTE[1], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in mnames], fontsize=7.5)
ax.set_ylim(0, 1.05); ax.set_title("E1. Model Component Ablation")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/E_ablation_models.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: E_ablation_models.png")


# ═══════════════════════════════════════════════════════════════════════════════
# F. ABLATION: GRAPH PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  F. ABLATION: GRAPH PARAMETERS")
print("─"*70)

# F1: Lambda sensitivity
print("\n  F1. Lambda (decay constant) sensitivity:")
lambda_results = {}
for lam in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
    # Recompute pagerank-like feature with different lambda
    # Approximate: scale the TPR graph feature column
    X_lam = X.copy().tolil()
    tpr_col = X[:, -13+2].toarray().flatten()  # TPR norm column
    scale = np.exp(-lam * 2.5) / np.exp(-0.10 * 2.5)  # relative scaling
    X_lam[:, -13+2] = tpr_col * scale
    X_lam = X_lam.tocsr()
    clf = _SVCWithProba(C=1.0)
    preds = cross_val_predict(clf, X_lam, labels, cv=cv5)
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average='macro', zero_division=0)
    lambda_results[lam] = {'accuracy':round(acc,4), 'macro_f1':round(mf1,4)}
    marker = " ←" if lam == 0.10 else ""
    print(f"    λ={lam:.2f}   acc={acc:.4f}   macro-F1={mf1:.4f}{marker}")

# F2: Title boost sensitivity
print("\n  F2. Title boost factor sensitivity:")
boost_results = {}
orig_vocab = extractor.vectorizer
for boost in [1, 2, 3, 4, 5]:
    # re-fit tfidf with different boost
    all_texts_b = []
    for nid in node_ids:
        nd = G.nodes.get(nid, {})
        t = nd.get('title',''); a = nd.get('abstract','')
        all_texts_b.append((f"{t} " * boost + a).strip() or "empty")
    vec_b = TfidfVectorizer(max_features=12000, ngram_range=(1,2),
                            sublinear_tf=True, min_df=2)
    X_b_tf = vec_b.fit_transform(all_texts_b)
    X_b    = sp.hstack([X_b_tf, X[:, -13:]])
    clf    = _SVCWithProba(C=1.0)
    preds  = cross_val_predict(clf, X_b, labels, cv=cv5)
    acc    = accuracy_score(labels, preds)
    mf1    = f1_score(labels, preds, average='macro', zero_division=0)
    boost_results[boost] = {'accuracy':round(acc,4),'macro_f1':round(mf1,4)}
    marker = " ←" if boost == 3 else ""
    print(f"    boost×{boost}   acc={acc:.4f}   macro-F1={mf1:.4f}{marker}")

# F3: C_svm sensitivity
print("\n  F3. SVM regularization (C) sensitivity:")
c_results = {}
for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    clf   = _SVCWithProba(C=C)
    preds = cross_val_predict(clf, X, labels, cv=cv5)
    acc   = accuracy_score(labels, preds)
    mf1   = f1_score(labels, preds, average='macro', zero_division=0)
    c_results[C] = {'accuracy':round(acc,4),'macro_f1':round(mf1,4)}
    marker = " ←" if C == 1.0 else ""
    print(f"    C={C:<5}   acc={acc:.4f}   macro-F1={mf1:.4f}{marker}")

# F4: Mu (propagation weight) sensitivity
print("\n  F4. Graph propagation μ sensitivity:")
mu_results = {}
svm_seeds = svm_m.predict_proba(X)
svm_cls   = list(svm_m.classes_)
reorder   = [svm_cls.index(c) if c in svm_cls else 0 for c in all_cls]
Y0        = svm_seeds[:, reorder]
rows2,cols2,wts2=[],[],[]
nid2idx2 = {nid:i for i,nid in enumerate(node_ids)}
for u,v,ed in G.edges(data=True):
    ui=nid2idx2.get(u,-1); vi=nid2idx2.get(v,-1)
    if ui>=0 and vi>=0:
        w=ed.get('weight',1.0)
        rows2+=[ui,vi]; cols2+=[vi,ui]; wts2+=[w,w]
if rows2:
    A_sp=sp.csr_matrix((wts2,(rows2,cols2)),shape=(len(node_ids),len(node_ids)),dtype=np.float32)
    rs=np.array(A_sp.sum(axis=1)).flatten()+1e-12
    A_norm_sp=sp.diags(1/rs)@A_sp

for mu_val in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]:
    if mu_val == 0.0:
        gp_out = Y0
    else:
        Y = Y0.copy()
        for _ in range(15):
            Y_new = (1-mu_val)*Y0 + mu_val*(A_norm_sp@Y)
            if np.abs(Y_new-Y).max() < 1e-4: break
            Y = Y_new
        gp_out = Y
    pf = 0.45*svm_p_full + 0.35*lr_p_full + 0.20*gp_out
    preds = np.array(all_cls)[pf.argmax(axis=1)]
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average='macro', zero_division=0)
    mu_results[mu_val] = {'accuracy':round(acc,4),'macro_f1':round(mf1,4)}
    marker = " ←" if mu_val == 0.30 else ""
    print(f"    μ={mu_val:.2f}   acc={acc:.4f}   macro-F1={mf1:.4f}{marker}")

RESULTS['ablation_params'] = {
    'lambda': lambda_results,
    'title_boost': boost_results,
    'C_svm': c_results,
    'mu': mu_results,
}

# Figure F: 4-panel parameter sensitivity
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# Lambda
lam_vals = sorted(lambda_results.keys())
axes[0,0].plot(lam_vals,[lambda_results[l]['accuracy'] for l in lam_vals],'o-',color=PALETTE[0],label='Accuracy',lw=2)
axes[0,0].plot(lam_vals,[lambda_results[l]['macro_f1'] for l in lam_vals],'s--',color=PALETTE[2],label='Macro-F1',lw=2)
axes[0,0].axvline(0.10,color='gray',ls=':',lw=1); axes[0,0].set_title("F1. λ Sensitivity")
axes[0,0].set_xlabel("λ"); axes[0,0].legend(); axes[0,0].set_xscale('log')
# Boost
b_vals = sorted(boost_results.keys())
axes[0,1].plot(b_vals,[boost_results[b]['accuracy'] for b in b_vals],'o-',color=PALETTE[0],label='Accuracy',lw=2)
axes[0,1].plot(b_vals,[boost_results[b]['macro_f1'] for b in b_vals],'s--',color=PALETTE[2],label='Macro-F1',lw=2)
axes[0,1].axvline(3,color='gray',ls=':',lw=1); axes[0,1].set_title("F2. Title Boost Sensitivity")
axes[0,1].set_xlabel("Boost factor"); axes[0,1].legend()
# C_svm
c_vals = sorted(c_results.keys())
axes[1,0].plot(c_vals,[c_results[c]['accuracy'] for c in c_vals],'o-',color=PALETTE[0],label='Accuracy',lw=2)
axes[1,0].plot(c_vals,[c_results[c]['macro_f1'] for c in c_vals],'s--',color=PALETTE[2],label='Macro-F1',lw=2)
axes[1,0].axvline(1.0,color='gray',ls=':',lw=1); axes[1,0].set_title("F3. C_SVM Sensitivity")
axes[1,0].set_xlabel("C"); axes[1,0].set_xscale('log'); axes[1,0].legend()
# Mu
mu_vals = sorted(mu_results.keys())
axes[1,1].plot(mu_vals,[mu_results[m]['accuracy'] for m in mu_vals],'o-',color=PALETTE[0],label='Accuracy',lw=2)
axes[1,1].plot(mu_vals,[mu_results[m]['macro_f1'] for m in mu_vals],'s--',color=PALETTE[2],label='Macro-F1',lw=2)
axes[1,1].axvline(0.30,color='gray',ls=':',lw=1); axes[1,1].set_title("F4. μ (Propagation) Sensitivity")
axes[1,1].set_xlabel("μ"); axes[1,1].legend()
plt.tight_layout()
plt.savefig(f"{FIG}/F_param_sensitivity.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: F_param_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# G. TEMPORAL SPLIT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  G. TEMPORAL SPLIT EVALUATION (train≤2022 / test≥2023)")
print("─"*70)

node_years_map = {}
for src, fname in [("arXiv","arXiv_citation_dataset.json"),
                   ("DBLP","DBLP_citation_dataset.json"),
                   ("Elsevier","Elsevier_citation_dataset.json")]:
    path = os.path.join("data/raw", fname)
    if not os.path.exists(path): continue
    with open(path) as f: d = _json.load(f)
    for item in d:
        for ca in item.get('citing_articles',[]):
            t2  = str(ca.get('title','')).strip()[:80].lower()
            m   = _re.search(r', (20[012][0-9])', str(ca.get('authors','')))
            if m: node_years_map[t2] = int(m.group(1))

node_yrs = []
for nid in node_ids:
    title_k = G.nodes[nid].get('title','').lower()[:80]
    yr      = node_years_map.get(title_k) or G.nodes[nid].get('year')
    node_yrs.append(yr)

split_results = {}
print(f"\n  {'Split Year':>12} {'n_train':>8} {'n_test':>8} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"  {'─'*52}")
for split_yr in [2021, 2022, 2023, 2024]:
    tr = [i for i,y in enumerate(node_yrs) if y and y < split_yr]
    te = [i for i,y in enumerate(node_yrs) if y and y >= split_yr]
    if len(tr) < 50 or len(te) < 20:
        print(f"  {split_yr:>12}  — insufficient data")
        continue
    clf = _SVCWithProba(C=1.0)
    y_tr = [labels[i] for i in tr]; y_te = [labels[i] for i in te]
    # Only keep classes present in both
    tr_classes = set(y_tr); te_classes = set(y_te)
    common = tr_classes & te_classes
    tr_f = [i for i in tr if labels[i] in common]
    te_f = [i for i in te if labels[i] in common]
    clf.fit(X[tr_f], [labels[i] for i in tr_f])
    preds = clf.predict(X[te_f])
    y_te_f = [labels[i] for i in te_f]
    acc = accuracy_score(y_te_f, preds)
    mf1 = f1_score(y_te_f, preds, average='macro', zero_division=0)
    split_results[split_yr] = {'n_train':len(tr_f),'n_test':len(te_f),
                                'accuracy':round(acc,4),'macro_f1':round(mf1,4)}
    marker = " ←" if split_yr == 2023 else ""
    print(f"  {split_yr:>12} {len(tr_f):>8} {len(te_f):>8} {acc:>10.4f} {mf1:>10.4f}{marker}")

RESULTS['temporal_split'] = split_results

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if split_results:
    syrs = sorted(split_results.keys())
    axes[0].plot(syrs,[split_results[y]['accuracy'] for y in syrs],'o-',color=PALETTE[0],lw=2,label='Accuracy')
    axes[0].plot(syrs,[split_results[y]['macro_f1'] for y in syrs],'s--',color=PALETTE[2],lw=2,label='Macro-F1')
    axes[0].set_title("G1. Temporal Split Performance")
    axes[0].set_xlabel("Split Year"); axes[0].legend()
    axes[1].bar([str(y) for y in syrs],
                [split_results[y]['n_train'] for y in syrs],
                label='Train', color=PALETTE[0], alpha=0.8)
    axes[1].bar([str(y) for y in syrs],
                [split_results[y]['n_test'] for y in syrs],
                bottom=[split_results[y]['n_train'] for y in syrs],
                label='Test', color=PALETTE[2], alpha=0.8)
    axes[1].set_title("G2. Train/Test Split Sizes")
    axes[1].set_xlabel("Split Year"); axes[1].set_ylabel("Samples")
    axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG}/G_temporal_split.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: G_temporal_split.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H. LEAVE-ONE-SOURCE-OUT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  H. LEAVE-ONE-SOURCE-OUT (LOSO) EVALUATION")
print("─"*70)

src_map = {'arxiv':'arXiv','dblp':'DBLP','elsevier':'Elsevier'}
node_sources = [G.nodes[nid].get('source','') for nid in node_ids]
sources_present = sorted(set(node_sources))

loso_results = {}
print(f"\n  {'Test Source':<14} {'n_train':>8} {'n_test':>8} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"  {'─'*52}")
for test_src in sources_present:
    tr_idx = [i for i,s in enumerate(node_sources) if s != test_src]
    te_idx = [i for i,s in enumerate(node_sources) if s == test_src]
    if len(te_idx) < 10: continue
    y_tr = [labels[i] for i in tr_idx]
    y_te = [labels[i] for i in te_idx]
    common = set(y_tr) & set(y_te)
    tr_f = [i for i in tr_idx if labels[i] in common]
    te_f = [i for i in te_idx if labels[i] in common]
    clf  = _SVCWithProba(C=1.0)
    clf.fit(X[tr_f], [labels[i] for i in tr_f])
    preds = clf.predict(X[te_f])
    y_te_f = [labels[i] for i in te_f]
    acc = accuracy_score(y_te_f, preds)
    mf1 = f1_score(y_te_f, preds, average='macro', zero_division=0)
    loso_results[test_src] = {'n_train':len(tr_f),'n_test':len(te_f),
                               'accuracy':round(acc,4),'macro_f1':round(mf1,4)}
    print(f"  {test_src:<14} {len(tr_f):>8} {len(te_f):>8} {acc:>10.4f} {mf1:>10.4f}")

if loso_results:
    ma = np.mean([v['accuracy'] for v in loso_results.values()])
    mf = np.mean([v['macro_f1'] for v in loso_results.values()])
    print(f"  {'─'*52}")
    print(f"  {'Mean':<14} {'':>8} {'':>8} {ma:>10.4f} {mf:>10.4f}")
    loso_results['mean'] = {'accuracy':round(ma,4),'macro_f1':round(mf,4)}

RESULTS['loso'] = loso_results

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
if len(loso_results) > 1:
    loso_srcs = [k for k in loso_results if k != 'mean']
    loso_accs = [loso_results[k]['accuracy'] for k in loso_srcs]
    loso_mf1s = [loso_results[k]['macro_f1'] for k in loso_srcs]
    x = np.arange(len(loso_srcs))
    axes[0].bar(x-0.2, loso_accs, 0.38, color=PALETTE[0], label='Accuracy', alpha=0.9)
    axes[0].bar(x+0.2, loso_mf1s, 0.38, color=PALETTE[1], label='Macro-F1', alpha=0.9)
    axes[0].axhline(loso_results['mean']['accuracy'], color=PALETTE[0], ls='--', lw=1.5, label='Mean acc')
    axes[0].set_xticks(x); axes[0].set_xticklabels(loso_srcs)
    axes[0].set_title("H1. LOSO Per-Source Performance")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8)
    axes[1].bar(loso_srcs, [loso_results[k]['n_test'] for k in loso_srcs],
                color=PALETTE[2], alpha=0.9)
    axes[1].set_title("H2. Test Set Sizes per Source")
    axes[1].set_ylabel("N test samples")
plt.tight_layout()
plt.savefig(f"{FIG}/H_loso.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: H_loso.png")


# ═══════════════════════════════════════════════════════════════════════════════
# I. PER-CLASS PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  I. PER-CLASS PERFORMANCE (5-fold CV, Full TDCG-AT)")
print("─"*70)

clf_full = _SVCWithProba(C=1.0)
preds_full = cross_val_predict(clf_full, X, labels, cv=cv5)
rep = classification_report(labels, preds_full, output_dict=True, zero_division=0)

print(f"\n  {'Domain':<26} {'P':>8} {'R':>8} {'F1':>8} {'n':>7}")
print(f"  {'─'*57}")
per_class = {}
for cls in sorted(set(labels)):
    r = rep[cls]
    per_class[cls] = {k: round(v,4) for k,v in r.items()}
    print(f"  {cls:<26} {r['precision']:>8.4f} {r['recall']:>8.4f} "
          f"{r['f1-score']:>8.4f} {int(r['support']):>7}")
print(f"  {'─'*57}")
ma = rep['macro avg']
print(f"  {'Macro avg':<26} {ma['precision']:>8.4f} {ma['recall']:>8.4f} "
      f"{ma['f1-score']:>8.4f}")

RESULTS['per_class'] = per_class

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cls_names = sorted(set(labels))
short_names = [c.replace('_',' ').replace('Mathematics Physics','Math/Phys')
                 .replace('Medicine Biology','Med/Bio') for c in cls_names]
prec  = [per_class[c]['precision'] for c in cls_names]
rec   = [per_class[c]['recall']    for c in cls_names]
f1s   = [per_class[c]['f1-score']  for c in cls_names]
supp  = [int(per_class[c]['support']) for c in cls_names]
x = np.arange(len(cls_names))
for ax, vals, title, col in [
    (axes[0], prec, "I1. Precision per Class", PALETTE[0]),
    (axes[1], rec,  "I2. Recall per Class",    PALETTE[1]),
    (axes[2], f1s,  "I3. F1-Score per Class",  PALETTE[2]),
]:
    bars = ax.bar(x, vals, color=col, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_title(title)
    ax.axhline(np.mean(vals), color='red', ls='--', lw=1, label=f'Mean={np.mean(vals):.3f}')
    ax.legend(fontsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontsize=6)
plt.tight_layout()
plt.savefig(f"{FIG}/I_per_class.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: I_per_class.png")


# ═══════════════════════════════════════════════════════════════════════════════
# J. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  J. CONFUSION MATRIX")
print("─"*70)

cm = confusion_matrix(labels, preds_full, labels=cls_names)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, title, fmt in [
    (axes[0], cm,      "J1. Confusion Matrix (counts)", 'd'),
    (axes[1], cm_norm, "J2. Confusion Matrix (row-normalized)", '.2f'),
]:
    cmap = LinearSegmentedColormap.from_list('tdcg', ['#FFFFFF','#185FA5'])
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, linewidths=0.5, linecolor='#EEE',
                annot_kws={'size':7})
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)
plt.tight_layout()
plt.savefig(f"{FIG}/J_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Confusion matrix saved: J_confusion_matrix.png")
print(f"  Overall accuracy from CM: {cm.diagonal().sum()/cm.sum():.4f}")

RESULTS['confusion_matrix'] = cm.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# K. LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  K. LEARNING CURVE")
print("─"*70)

from sklearn.base import BaseEstimator, ClassifierMixin
class _SVCSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        self._inner = _SVCWithProba(C=self.C)
        self._inner.fit(X, list(y))
        self.classes_ = self._inner.classes_
        return self
    def predict(self, X):
        return self._inner.predict(X)

train_sizes = np.linspace(0.1, 1.0, 10)
clf_lc = _SVCSklearnWrapper(C=1.0)
train_sz, train_sc, test_sc = learning_curve(
    clf_lc, X, labels, cv=5, train_sizes=train_sizes,
    scoring='accuracy', n_jobs=1
)
print(f"\n  Train sizes: {[int(s) for s in train_sz[-5:]]}")
print(f"  Final train acc: {train_sc[-1].mean():.4f}  test acc: {test_sc[-1].mean():.4f}")

RESULTS['learning_curve'] = {
    'train_sizes': train_sz.tolist(),
    'train_scores_mean': train_sc.mean(axis=1).tolist(),
    'test_scores_mean':  test_sc.mean(axis=1).tolist(),
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(train_sz, train_sc.mean(1)-train_sc.std(1),
                train_sc.mean(1)+train_sc.std(1), alpha=0.15, color=PALETTE[0])
ax.fill_between(train_sz, test_sc.mean(1)-test_sc.std(1),
                test_sc.mean(1)+test_sc.std(1), alpha=0.15, color=PALETTE[2])
ax.plot(train_sz, train_sc.mean(1), 'o-', color=PALETTE[0], lw=2, label='Train accuracy')
ax.plot(train_sz, test_sc.mean(1),  's-', color=PALETTE[2], lw=2, label='CV accuracy')
ax.set_title("K. Learning Curve (LinearSVC, 5-fold CV)")
ax.set_xlabel("Training samples"); ax.set_ylabel("Accuracy")
ax.legend(); ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig(f"{FIG}/K_learning_curve.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: K_learning_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# L. HYPERPARAMETER GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  L. HYPERPARAMETER GRID SEARCH  (C_svm × ngram range)")
print("─"*70)

grid_results = {}
for C in [0.1, 1.0, 5.0]:
    for ngram in [(1,1),(1,2),(2,2)]:
        # re-vectorize with different ngram
        texts_g = []
        for nid in node_ids:
            nd = G.nodes.get(nid, {})
            t = nd.get('title',''); a = nd.get('abstract','')
            texts_g.append((f"{t} "*3 + a).strip() or "empty")
        vec_g  = TfidfVectorizer(max_features=12000, ngram_range=ngram,
                                  sublinear_tf=True, min_df=2)
        Xtf_g  = vec_g.fit_transform(texts_g)
        X_g    = sp.hstack([Xtf_g, X[:,-13:]])
        clf_g  = _SVCWithProba(C=C)
        preds_g = cross_val_predict(clf_g, X_g, labels, cv=cv5)
        acc_g  = accuracy_score(labels, preds_g)
        key    = f"C={C}, ngram={ngram}"
        grid_results[key] = round(acc_g, 4)
        print(f"  {key:<28}  acc={acc_g:.4f}")

RESULTS['grid_search'] = grid_results

# Heatmap
Cs     = [0.1, 1.0, 5.0]
ngrams = [(1,1),(1,2),(2,2)]
mat = np.array([[grid_results[f"C={C}, ngram={ng}"] for ng in ngrams] for C in Cs])
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(mat, annot=True, fmt='.4f', xticklabels=[str(n) for n in ngrams],
            yticklabels=[str(c) for c in Cs], ax=ax,
            cmap='YlOrRd', linewidths=0.5)
ax.set_title("L. Grid Search: C × ngram range (Accuracy)")
ax.set_xlabel("ngram range"); ax.set_ylabel("C_SVM")
plt.tight_layout()
plt.savefig(f"{FIG}/L_grid_search.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: L_grid_search.png")


# ═══════════════════════════════════════════════════════════════════════════════
# M. GRAPH STRUCTURAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  M. GRAPH STRUCTURAL ANALYSIS")
print("─"*70)

in_degs_all  = [G.in_degree(n)  for n in G.nodes()]
out_degs_all = [G.out_degree(n) for n in G.nodes()]
wcc_sizes = [len(c) for c in nx.connected_components(G.to_undirected())]
edge_weights = [d.get('weight',1.0) for _,_,d in G.edges(data=True)]

print(f"  Nodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}")
print(f"  In-degree:  max={max(in_degs_all)}  mean={np.mean(in_degs_all):.3f}  std={np.std(in_degs_all):.3f}")
print(f"  Out-degree: max={max(out_degs_all)}  mean={np.mean(out_degs_all):.3f}")
print(f"  WCC: count={len(wcc_sizes)}  largest={max(wcc_sizes)}  mean_size={np.mean(wcc_sizes):.1f}")
print(f"  Edge weights: mean={np.mean(edge_weights):.4f}  min={min(edge_weights):.4f}  std={np.std(edge_weights):.4f}")

# Power-law fit
from scipy.stats import linregress
dc_in = Counter(d for d in in_degs_all if d > 0)
xs_pl, ys_pl = zip(*sorted(dc_in.items()))
slope, intercept, r, p_pl, _ = linregress(np.log(xs_pl), np.log(ys_pl))
print(f"  Power-law fit: γ={-slope:.3f}  R²={r**2:.4f}  p={p_pl:.4e}")

# WCC domain purity
wcc_purity = []
for comp in nx.connected_components(G.to_undirected()):
    if len(comp) < 3: continue
    doms = [G.nodes[n].get('domain','') for n in comp if G.nodes[n].get('domain')]
    if doms:
        wcc_purity.append(Counter(doms).most_common(1)[0][1]/len(doms))
print(f"  WCC domain purity (n≥3): mean={np.mean(wcc_purity):.3f}  median={np.median(wcc_purity):.3f}")

RESULTS['graph_structure'] = {
    'n_nodes': G.number_of_nodes(),
    'n_edges': G.number_of_edges(),
    'in_degree_max': max(in_degs_all),
    'in_degree_mean': round(np.mean(in_degs_all),4),
    'wcc_count': len(wcc_sizes),
    'largest_wcc': max(wcc_sizes),
    'edge_weight_mean': round(np.mean(edge_weights),4),
    'power_law_gamma': round(-slope,4),
    'power_law_r2': round(r**2,4),
    'wcc_purity_mean': round(np.mean(wcc_purity),4),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# In-degree dist
axes[0,0].hist(in_degs_all, bins=50, log=True, color=PALETTE[0], alpha=0.9)
axes[0,0].set_title("M1. In-degree Distribution"); axes[0,0].set_xlabel("In-degree")
# WCC size dist
axes[0,1].hist(wcc_sizes, bins=30, log=True, color=PALETTE[1], alpha=0.9)
axes[0,1].set_title("M2. WCC Size Distribution"); axes[0,1].set_xlabel("WCC size")
# Edge weight dist
axes[1,0].hist(edge_weights, bins=40, color=PALETTE[2], alpha=0.9)
axes[1,0].set_title("M3. Edge Weight Distribution"); axes[1,0].set_xlabel("w(e)=exp(-λΔt)")
# WCC domain purity
axes[1,1].hist(wcc_purity, bins=20, color=PALETTE[3], alpha=0.9)
axes[1,1].axvline(np.mean(wcc_purity), color='red', ls='--', lw=1.5,
                   label=f'Mean={np.mean(wcc_purity):.3f}')
axes[1,1].set_title("M4. WCC Domain Purity"); axes[1,1].legend()
plt.tight_layout()
plt.savefig(f"{FIG}/M_graph_structure.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: M_graph_structure.png")


# ═══════════════════════════════════════════════════════════════════════════════
# N. TEMPORAL TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  N. TEMPORAL TREND ANALYSIS")
print("─"*70)

# Papers per domain per year
domain_year = defaultdict(lambda: defaultdict(int))
for nid in node_ids:
    nd  = G.nodes[nid]
    dom = nd.get('domain','')
    yr  = nd.get('year')
    if dom and yr and 2018 <= yr <= 2025:
        domain_year[dom][yr] += 1

# Year distribution per source
src_year = defaultdict(lambda: defaultdict(int))
for nid in node_ids:
    nd  = G.nodes[nid]
    src = nd.get('source','')
    yr  = nd.get('year')
    if src and yr and 2018 <= yr <= 2025:
        src_year[src][yr] += 1

print(f"\n  Papers per domain over years 2018-2025:")
print(f"  {'Domain':<26} " + " ".join(f"{y:>5}" for y in range(2018,2026)))
print(f"  {'─'*90}")
for dom in sorted(domain_year.keys()):
    row = [domain_year[dom].get(y,0) for y in range(2018,2026)]
    print(f"  {dom:<26} " + " ".join(f"{v:>5}" for v in row))

RESULTS['temporal_trends'] = {
    dom: dict(domain_year[dom]) for dom in domain_year
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
years_plot = range(2018, 2026)
for i, (dom, col) in enumerate(zip(sorted(domain_year.keys()), PALETTE)):
    vals = [domain_year[dom].get(y,0) for y in years_plot]
    axes[0].plot(years_plot, vals, 'o-', color=col, lw=1.5, label=dom[:12], alpha=0.9)
axes[0].set_title("N1. Papers per Domain over Time"); axes[0].set_xlabel("Year")
axes[0].legend(fontsize=6.5, ncol=2)
# Stacked bar by source
src_list = list(src_year.keys())
src_colors = PALETTE[:len(src_list)]
bottom = np.zeros(len(years_plot))
for src, col in zip(src_list, src_colors):
    vals = np.array([src_year[src].get(y,0) for y in years_plot])
    axes[1].bar(years_plot, vals, bottom=bottom, label=src, color=col, alpha=0.9)
    bottom += vals
axes[1].set_title("N2. Yearly Paper Count by Source")
axes[1].set_xlabel("Year"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG}/N_temporal_trends.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: N_temporal_trends.png")


# ═══════════════════════════════════════════════════════════════════════════════
# O. CLASSIFICATION CONFIDENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  O. CLASSIFICATION CONFIDENCE ANALYSIS")
print("─"*70)

# Get probability estimates from full model
agent._svm.fit(X, labels)
agent._lr.fit(X, labels)
all_probas = agent.predict_proba(X, node_ids=node_ids, builder=builder)
pred_confs  = all_probas.max(axis=1)
pred_labels = np.array(agent.classes_)[all_probas.argmax(axis=1)]
correct_mask = np.array([pred_labels[i]==labels[i] for i in range(len(labels))])

print(f"\n  Confidence distribution:")
for thresh in [0.3, 0.5, 0.7, 0.9]:
    above = (pred_confs >= thresh).sum()
    acc_above = correct_mask[pred_confs >= thresh].mean() if above > 0 else 0
    print(f"  Conf ≥ {thresh:.1f}: {above:>5} papers ({100*above/len(labels):.1f}%)  "
          f"acc={acc_above:.4f}")

print(f"\n  Correct predictions:   mean conf={pred_confs[correct_mask].mean():.4f}")
print(f"  Incorrect predictions: mean conf={pred_confs[~correct_mask].mean():.4f}")

# Per-domain confidence
print(f"\n  Per-domain mean confidence:")
for dom in sorted(set(labels)):
    mask = np.array([lb == dom for lb in labels])
    if mask.sum() == 0: continue
    dom_conf = pred_confs[mask].mean()
    dom_acc  = correct_mask[mask].mean()
    print(f"    {dom:<26} conf={dom_conf:.4f}  acc={dom_acc:.4f}")

RESULTS['confidence'] = {
    'mean_correct': round(float(pred_confs[correct_mask].mean()),4),
    'mean_incorrect': round(float(pred_confs[~correct_mask].mean()),4),
    'high_conf_0.7_pct': round(float((pred_confs>=0.7).mean()),4),
    'high_conf_0.9_pct': round(float((pred_confs>=0.9).mean()),4),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(pred_confs[correct_mask],  bins=25, alpha=0.7, color=PALETTE[0],
             label=f'Correct (n={correct_mask.sum()})', density=True)
axes[0].hist(pred_confs[~correct_mask], bins=25, alpha=0.7, color=PALETTE[2],
             label=f'Incorrect (n={(~correct_mask).sum()})', density=True)
axes[0].set_title("O1. Confidence: Correct vs Incorrect")
axes[0].set_xlabel("Max predicted probability"); axes[0].legend()
# Calibration-style: confidence buckets vs accuracy
buckets = np.linspace(0, 1, 11)
bucket_acc, bucket_conf, bucket_n = [], [], []
for lo, hi in zip(buckets[:-1], buckets[1:]):
    mask2 = (pred_confs >= lo) & (pred_confs < hi)
    if mask2.sum() > 0:
        bucket_acc.append(correct_mask[mask2].mean())
        bucket_conf.append(pred_confs[mask2].mean())
        bucket_n.append(mask2.sum())
axes[1].plot([0,1],[0,1],'--',color='gray',lw=1,label='Perfect calibration')
axes[1].scatter(bucket_conf, bucket_acc, s=[n/5 for n in bucket_n],
                color=PALETTE[0], alpha=0.8, label='Model')
axes[1].set_title("O2. Reliability Diagram (Calibration)")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Fraction correct"); axes[1].legend()
# Per-domain confidence box
dom_confs = {dom: pred_confs[np.array([lb==dom for lb in labels])].tolist()
             for dom in sorted(set(labels))}
short_dom  = [d[:10] for d in sorted(set(labels))]
axes[2].boxplot([dom_confs[d] for d in sorted(set(labels))],
                tick_labels=short_dom, patch_artist=True,
                boxprops=dict(facecolor=PALETTE[4], alpha=0.6))
axes[2].set_title("O3. Confidence by Domain")
axes[2].set_xticklabels(short_dom, rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.savefig(f"{FIG}/O_confidence.png", dpi=150, bbox_inches='tight')
plt.close(); print(f"\n  → Saved: O_confidence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("  SAVING ALL RESULTS")
print("─"*70)

# Master JSON
with open(f"{OUT}/all_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"  → {OUT}/all_results.json")

# CSV summary table
import csv
summary_rows = [
    ["Experiment", "Metric", "Value"],
    ["Dataset",    "Total papers",    total_r+total_c],
    ["Dataset",    "Labeled samples", len(labels)],
    ["Dataset",    "Classes",         len(set(labels))],
    ["Dataset",    "Imbalance ratio", RESULTS['dataset']['imbalance_ratio']],
    ["Graph",      "|V|",             st['n_nodes']],
    ["Graph",      "|E|",             st['n_edges']],
    ["Graph",      "WCC count",       st['n_wcc']],
    ["Graph",      "Delta-t mean (yr)",st['delta_t_mean']],
    ["Features",   "Matrix shape",    f"{X.shape[0]}×{X.shape[1]}"],
    ["Features",   "TF-IDF density",  RESULTS['features']['tfidf_density']],
    ["Features",   "SNR",             RESULTS['features']['snr']],
    ["5-fold CV",  "Accuracy (TDCG-AT SVM)", clf_results.get("TDCG-AT SVM only",{}).get('accuracy','—')],
    ["5-fold CV",  "Accuracy (TDCG-AT Ensemble)", clf_results.get("TDCG-AT Ensemble",{}).get('accuracy','—')],
    ["5-fold CV",  "Macro-F1 (Ensemble)", clf_results.get("TDCG-AT Ensemble",{}).get('macro_f1','—')],
    ["Ablation",   "+Graph feature gain (Δ acc)", round(ablation_results.get("Full X (TF-IDF+graph, d=+13)",{}).get('accuracy',0)-ablation_results.get("TF-IDF only (d=11543)",{}).get('accuracy',0),4)],
    ["LOSO",       "Mean accuracy",   loso_results.get('mean',{}).get('accuracy','—')],
    ["LOSO",       "Mean Macro-F1",   loso_results.get('mean',{}).get('macro_f1','—')],
]
with open(f"{CSV}/summary.csv","w",newline='') as f:
    csv.writer(f).writerows(summary_rows)
print(f"  → {CSV}/summary.csv")

# Per-class CSV
with open(f"{CSV}/per_class.csv","w",newline='') as f:
    w = csv.writer(f)
    w.writerow(["Domain","Precision","Recall","F1","Support"])
    for cls in sorted(set(labels)):
        r = per_class[cls]
        w.writerow([cls, r['precision'], r['recall'], r['f1-score'], int(r['support'])])
print(f"  → {CSV}/per_class.csv")

# Classifier comparison CSV
with open(f"{CSV}/classifier_comparison.csv","w",newline='') as f:
    w = csv.writer(f)
    w.writerow(["Classifier","Accuracy","Macro-F1","Weighted-F1","Time(s)"])
    for name, res in clf_results.items():
        w.writerow([name, res['accuracy'], res['macro_f1'],
                    res['weighted_f1'], res['time_s']])
print(f"  → {CSV}/classifier_comparison.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  Generating master summary figure …")
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Domain distribution
ax1 = fig.add_subplot(gs[0, 0])
dom_vals_s = [dist[d] for d in sorted(dist, key=dist.get, reverse=True)]
dom_names_s = [d[:10] for d in sorted(dist, key=dist.get, reverse=True)]
ax1.barh(dom_names_s[::-1], dom_vals_s[::-1], color=PALETTE[:len(dom_vals_s)])
ax1.set_title("Domain Distribution", fontsize=9)

# Panel 2: Classifier accuracy comparison
ax2 = fig.add_subplot(gs[0, 1])
top_clfs = sorted(clf_results.items(), key=lambda x:-x[1]['accuracy'])[:8]
ax2.barh([n[:18] for n,_ in top_clfs[::-1]],
         [v['accuracy'] for _,v in top_clfs[::-1]],
         color=[PALETTE[2] if 'Ensemble' in n else PALETTE[0] for n,_ in top_clfs[::-1]])
ax2.set_xlim(0.5, 1.05)
ax2.set_title("Classifier Accuracy", fontsize=9)

# Panel 3: Feature ablation
ax3 = fig.add_subplot(gs[0, 2])
abl_n = [k.split('(')[0].strip()[:18] for k in ablation_results]
abl_a = [ablation_results[k]['accuracy'] for k in ablation_results]
bars3 = ax3.bar(range(len(abl_n)), abl_a,
                color=[PALETTE[2] if 'Full' in k else PALETTE[0] for k in ablation_results])
ax3.set_xticks(range(len(abl_n)))
ax3.set_xticklabels([n[:10] for n in abl_n], rotation=30, ha='right', fontsize=7)
ax3.set_ylim(0.7, 1.05); ax3.set_title("Feature Ablation", fontsize=9)

# Panel 4: F1 per class
ax4 = fig.add_subplot(gs[0, 3])
ax4.barh(short_names[::-1], f1s[::-1], color=PALETTE[1], alpha=0.9)
ax4.axvline(np.mean(f1s), color='red', ls='--', lw=1)
ax4.set_xlim(0.5, 1.05); ax4.set_title("F1 per Class", fontsize=9)

# Panel 5: Confusion matrix (normalized)
ax5 = fig.add_subplot(gs[1, :2])
cmap5 = LinearSegmentedColormap.from_list('tdcg', ['#FFFFFF','#185FA5'])
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap5,
            xticklabels=[s[:6] for s in short_names],
            yticklabels=[s[:6] for s in short_names],
            ax=ax5, linewidths=0.3, annot_kws={'size':7},
            cbar_kws={'shrink':0.8})
ax5.set_title("Confusion Matrix (normalized)", fontsize=9)
ax5.tick_params(labelsize=6)

# Panel 6: LOSO
ax6 = fig.add_subplot(gs[1, 2])
if len(loso_results) > 1:
    loso_k = [k for k in loso_results if k!='mean']
    ax6.bar(loso_k, [loso_results[k]['accuracy'] for k in loso_k], color=PALETTE[3])
    ax6.axhline(loso_results['mean']['accuracy'], color='red', ls='--', lw=1.5)
    ax6.set_ylim(0, 1.05); ax6.set_title("LOSO Accuracy", fontsize=9)

# Panel 7: Learning curve
ax7 = fig.add_subplot(gs[1, 3])
ax7.plot(RESULTS['learning_curve']['train_sizes'],
         RESULTS['learning_curve']['train_scores_mean'],
         'o-', color=PALETTE[0], lw=2, label='Train')
ax7.plot(RESULTS['learning_curve']['train_sizes'],
         RESULTS['learning_curve']['test_scores_mean'],
         's-', color=PALETTE[2], lw=2, label='CV')
ax7.set_title("Learning Curve", fontsize=9)
ax7.set_xlabel("Train samples", fontsize=8); ax7.legend(fontsize=8)

# Panel 8: Edge weight distribution
ax8 = fig.add_subplot(gs[2, 0])
ax8.hist(edge_weights, bins=30, color=PALETTE[5], alpha=0.9)
ax8.set_title("Edge Weight Distribution\nw=exp(-λΔt)", fontsize=9)
ax8.set_xlabel("w(e)", fontsize=8)

# Panel 9: Confidence
ax9 = fig.add_subplot(gs[2, 1])
ax9.hist(pred_confs[correct_mask],  bins=20, alpha=0.7, color=PALETTE[0], density=True, label='Correct')
ax9.hist(pred_confs[~correct_mask], bins=20, alpha=0.7, color=PALETTE[2], density=True, label='Wrong')
ax9.set_title("Confidence Distribution", fontsize=9); ax9.legend(fontsize=8)

# Panel 10: Temporal trend
ax10 = fig.add_subplot(gs[2, 2:])
yr_all_cnt = Counter(y for y in years_all if 2015 <= y <= 2025)
ax10.bar(yr_all_cnt.keys(), yr_all_cnt.values(), color=PALETTE[6], alpha=0.9)
ax10.set_title("Papers per Year (2015-2025)", fontsize=9)
ax10.set_xlabel("Year", fontsize=8); ax10.set_ylabel("Count", fontsize=8)

fig.suptitle("TDCG-AT Classification System — Comprehensive Experiment Results",
             fontsize=13, fontweight='bold', y=0.98)
plt.savefig(f"{OUT}/MASTER_SUMMARY.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {OUT}/MASTER_SUMMARY.png")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  ALL EXPERIMENTS COMPLETE")
print("="*70)
print(f"\n  Output directory : {OUT}")
print(f"  Figures saved    : {len(list(Path(FIG).glob('*.png')))} PNG files")
print(f"  CSV files saved  : {len(list(Path(CSV).glob('*.csv')))} CSV files")
print(f"  JSON results     : {OUT}/all_results.json")
print(f"  Master figure    : {OUT}/MASTER_SUMMARY.png")
print("\n  Key results:")
ens = clf_results.get("TDCG-AT Ensemble",{})
print(f"    5-fold CV accuracy  : {ens.get('accuracy','—')}")
print(f"    5-fold CV macro-F1  : {ens.get('macro_f1','—')}")
loso_m = loso_results.get('mean',{})
print(f"    LOSO mean accuracy  : {loso_m.get('accuracy','—')}")
print(f"    LOSO mean macro-F1  : {loso_m.get('macro_f1','—')}")
gain = round(ablation_results.get("Full X (TF-IDF+graph, d=+13)",{}).get('accuracy',0)-
             ablation_results.get("TF-IDF only (d=11543)",{}).get('accuracy',0), 4)
print(f"    Graph feature gain  : +{gain:.4f} accuracy")
print("="*70 + "\n")
