"""Évaluation de CN et Adamic-Adar sur le réseau de collaborations hep-th.


Protocole : apprentissage 1994–1996, test 1997–1999, seuils 3/3.
On prédit k = nombre de nouvelles collaborations (article [2]).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from src.datasets import load_hepth
from src.evaluate import precision_at, random_precision
from src.measures.local import aa_scores, cn_scores, rank
from src.split import temporal_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results" / "evaluation_cn_aa_hepth"


def at_k_with_ties(ranked, positives, k):
    """Diagnostic des ex aequo à la frontière du top-k.

    Retourne le nombre de bonnes prédictions pour le tri stable existant,
    ainsi que le nombre moyen attendu si les ex aequo au seuil étaient
    départagés uniformément au hasard (calcul analytique, sans simulation).
    """
    chosen = ranked[:k]
    hits = sum(pair in positives for pair, _ in chosen)
    if len(ranked) < k:
        return hits, float(hits), 0, 0

    threshold = ranked[k - 1][1]
    above = [(pair, score) for pair, score in ranked if score > threshold]
    tied = [pair for pair, score in ranked if score == threshold]
    places = k - len(above)
    hits_above = sum(pair in positives for pair, _ in above)
    hits_tied = sum(pair in positives for pair in tied)
    expected_hits = hits_above + places * hits_tied / len(tied)
    return hits, expected_hits, len(tied), places


def save_predictions(path, ranked, positives, k):
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rang", "sommet_u", "sommet_v", "score", "relation_future_retrouvee"])
        for i, (pair, score) in enumerate(ranked[:k], 1):
            wr.writerow([i, pair[0], pair[1], repr(score), int(pair in positives)])


def main():
    for filename in ("hep-th-abs.tar.gz", "hep-th-slacdates.tar.gz"):
        if not (DATA / filename).is_file():
            raise FileNotFoundError(f"Archive introuvable : {DATA / filename}")

    print("Chargement des donnees hep-th...")
    papers = load_hepth(DATA / "hep-th-abs.tar.gz", DATA / "hep-th-slacdates.tar.gz")
    s = temporal_split(papers, train=(1994, 1996), test=(1997, 1999),
                       k_train=3, k_test=3)
    k = len(s.new_edges)
    if not k:
        raise ValueError("Aucune nouvelle collaboration a retrouver")
    if k > s.num_candidates:
        raise ValueError("Plus de liens reels que de paires candidates")

    baseline = random_precision(k, s.num_candidates)
    print(f"Auteurs coeur : {len(s.core):,}")
    print(f"Paires candidates : {s.num_candidates:,}")
    print(f"Nouvelles collaborations : {k:,} => k={k:,}")
    print(f"Precision aleatoire attendue : {baseline:.6%}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, fn in (("CN", cn_scores), ("AA", aa_scores)):
        start = time.perf_counter()
        scores = fn(s.train_adj, nodes=s.core)
        ranked = rank(scores)
        elapsed = time.perf_counter() - start

        # Les scores positifs doivent couvrir au moins k candidats ici.
        # Sinon, compléter le classement avec des scores nuls exige une
        # convention supplémentaire qu'il faut documenter.
        if len(ranked) < k:
            raise RuntimeError(
                f"{name}: seulement {len(ranked)} candidats notes pour k={k}. "
                "Preciser comment traiter les scores nuls avant d'evaluer."
            )

        # Vérification du protocole et de la sortie des mesures.
        assert all(u in s.core and v in s.core for (u, v) in scores)
        assert all(v not in s.train_adj[u] for (u, v) in scores)
        hits, expected, n_tied, selected_tied = at_k_with_ties(ranked, s.new_edges, k)
        precision = precision_at(ranked, s.new_edges, k)
        assert abs(precision - hits / k) < 1e-12
        factor = precision / baseline
        outfile = OUT / f"top_{k}_{name.lower()}.csv"
        save_predictions(outfile, ranked, s.new_edges, k)

        print(f"\n===== {name} =====")
        print(f"Paires avec score > 0 : {len(ranked):,}")
        print(f"Temps calcul + tri : {elapsed:.4f} s")
        print(f"Bonnes predictions : {hits}/{k}")
        print(f"Precision@{k} : {precision:.4%}")
        print(f"Rappel@{k} : {hits / k:.4%}")
        print(f"Facteur vs hasard : x{factor:.2f}")
        print(f"Ex aequo au seuil : {n_tied:,} paires pour {selected_tied} places")
        print(f"Bonnes predictions attendues avec ex aequo aleatoires : {expected:.2f}")
        print(f"CSV : {outfile.relative_to(ROOT)}")
        summary.append({
            "algorithme": name,
            "candidats_scores_positifs": len(ranked),
            "k": k,
            "bonnes_predictions": hits,
            "precision_at_k": f"{precision:.12f}",
            "rappel_at_k": f"{hits / k:.12f}",
            "precision_aleatoire": f"{baseline:.12f}",
            "facteur_vs_hasard": f"{factor:.12f}",
            "ex_aequo_au_seuil": n_tied,
            "places_dans_ex_aequo": selected_tied,
            "bonnes_predictions_esperees_avec_ex_aequo_aleatoires": f"{expected:.6f}",
            "secondes_calcul_et_tri": f"{elapsed:.6f}",
        })

    with (OUT / "resume_cn_aa.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nTermine. Fichiers disponibles dans results/evaluation_cn_aa_hepth/.")



if __name__ == "__main__":
    main()
