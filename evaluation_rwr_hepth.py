"""Évalue RWR symétrisé sur hep-th avec le protocole temporel de l'article [2].


Protocole : apprentissage 1994–1996, test 1997–1999, cœur 3/3,
k = nombre de nouvelles collaborations. Le RWR existant calcule
score(u,v) = r_u(v) + r_v(u), adaptation symétrisée de rooted PageRank.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.datasets import load_hepth
from src.evaluate import precision_at, random_precision
from src.measures.global_measures import rwr_scores
from src.measures.local import rank
from src.split import temporal_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results" / "evaluation_rwr_hepth"


def ties_at_k(ranked, positives, k):
    """Nombre de paires ex æquo et espérance de bonnes prédictions à k."""
    if len(ranked) < k:
        raise ValueError("Moins de k paires classées : définir la gestion des scores nuls.")
    threshold = ranked[k - 1][1]
    above = 0
    hits_above = 0
    tied = 0
    hits_tied = 0
    for pair, score in ranked:
        if score > threshold:
            above += 1
            hits_above += pair in positives
        elif score == threshold:
            tied += 1
            hits_tied += pair in positives
        else:
            break
    places = k - above
    expected = hits_above + places * hits_tied / tied
    return tied, places, expected


def main():
    parser = argparse.ArgumentParser(description="Évaluation RWR (hep-th, article [2]).")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Probabilité de redémarrage (défaut : 0.15).")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="Seuil de convergence (défaut : 1e-10).")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Nombre maximal d'itérations (défaut : 1000).")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Nombre de sources calculées par bloc (défaut : 64).")
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.block_size <= 0 or args.max_iter <= 0 or args.tol <= 0:
        parser.error("alpha dans ]0,1[, tol > 0, max-iter et block-size > 0")

    for filename in ("hep-th-abs.tar.gz", "hep-th-slacdates.tar.gz"):
        if not (DATA / filename).is_file():
            raise FileNotFoundError(f"Archive absente : {DATA / filename}")

    print("Chargement des donnees hep-th...", flush=True)
    papers = load_hepth(DATA / "hep-th-abs.tar.gz", DATA / "hep-th-slacdates.tar.gz")
    s = temporal_split(papers, train=(1994, 1996), test=(1997, 1999),
                       k_train=3, k_test=3)
    k = len(s.new_edges)
    baseline = random_precision(k, s.num_candidates)
    print(f"Auteurs du coeur : {len(s.core):,}")
    print(f"Nouvelles collaborations : {k:,} => k={k:,}")
    print(f"Paires candidates : {s.num_candidates:,}")
    print(f"Hasard attendu : {baseline:.6%}")
    print(f"RWR symetrise : alpha={args.alpha}, tol={args.tol}, "
          f"max_iter={args.max_iter}, block_size={args.block_size}", flush=True)
    print("Calcul RWR complet (tous les auteurs du coeur)...", flush=True)

    start = time.perf_counter()
    scores = rwr_scores(s.train_adj, nodes=sorted(s.core),
                        alpha=args.alpha, tol=args.tol,
                        max_iter=args.max_iter, block_size=args.block_size)
    ranked = rank(scores)
    elapsed = time.perf_counter() - start

    if len(ranked) < k:
        raise RuntimeError(f"Seulement {len(ranked)} scores positifs pour k={k}.")
    assert all(u in s.core and v in s.core and v not in s.train_adj[u]
               for (u, v) in scores)
    hits = sum(pair in s.new_edges for pair, _ in ranked[:k])
    precision = precision_at(ranked, s.new_edges, k)
    recall = hits / k
    assert abs(precision - hits / k) < 1e-12
    tied, places, expected_hits = ties_at_k(ranked, s.new_edges, k)

    OUT.mkdir(parents=True, exist_ok=True)
    predictions_file = OUT / f"top_{k}_rwr.csv"
    with predictions_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rang", "sommet_u", "sommet_v", "score_rwr_symetrise",
                         "relation_future_retrouvee"])
        for idx, (pair, score) in enumerate(ranked[:k], start=1):
            writer.writerow([idx, pair[0], pair[1], repr(score), int(pair in s.new_edges)])

    summary_file = OUT / "resume_rwr.csv"
    row = {
        "algorithme": "RWR_symetrise", "alpha": args.alpha,
        "tol": args.tol, "max_iter": args.max_iter, "block_size": args.block_size,
        "auteurs_coeur": len(s.core), "paires_candidates": s.num_candidates,
        "nouvelles_relations": k, "paires_scores_positifs": len(ranked),
        "bonnes_predictions": hits, "precision_at_k": precision,
        "rappel_at_k": recall, "hasard": baseline,
        "facteur_vs_hasard": precision / baseline,
        "ex_aequo_au_seuil": tied, "places_dans_ex_aequo": places,
        "bonnes_predictions_esperees_ex_aequo_aleatoires": expected_hits,
        "temps_calcul_et_tri_secondes": elapsed,
    }
    with summary_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print("\n===== RWR =====")
    print(f"Paires avec score > 0 : {len(ranked):,}")
    print(f"Temps calcul + tri : {elapsed:.4f} s")
    print(f"Bonnes predictions : {hits}/{k}")
    print(f"Precision@{k} : {precision:.4%}")
    print(f"Rappel@{k} : {recall:.4%}")
    print(f"Facteur vs hasard : x{precision / baseline:.2f}")
    print(f"Ex aequo au seuil : {tied} paires pour {places} places")
    print(f"Bonnes predictions attendues avec ex aequo aleatoires : {expected_hits:.2f}")
    print(f"CSV : {predictions_file.relative_to(ROOT)}")
    print(f"Resume : {summary_file.relative_to(ROOT)}")
    print("Termine sans modification des algorithmes.")


if __name__ == "__main__":
    main()
