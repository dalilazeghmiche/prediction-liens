"""Compare rooted PageRank directionnel et RWR symétrisé sur hep-th.

Depuis la racine du dépôt (Windows PowerShell) :
    .\\.venv\\Scripts\\python.exe comparaison_rwr_original_symetrise_hepth.py

Dépendances : numpy, scipy, matplotlib (déjà utilisés dans le projet).

"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.datasets import load_hepth
from src.measures.global_measures import rwr_matrix
from src.split import temporal_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
OUTPUT = ROOT / "results" / "comparaison_rwr_original_symetrise_hepth"


def expected_top_k(sorted_scores, ordered_true, k):
    """Espérance exacte des vrais positifs avec départage uniforme des ex æquo."""
    v = sorted_scores
    threshold = v[k - 1]
    # -v est trié par ordre croissant.
    start = int(np.searchsorted(-v, -threshold, side="left"))
    end = int(np.searchsorted(-v, -threshold, side="right"))
    hits_above = int(np.count_nonzero(ordered_true[:start]))
    hits_tied = int(np.count_nonzero(ordered_true[start:end]))
    places = k - start
    return (hits_above + places * hits_tied / (end - start), end - start, places)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Comparaison du RWR directionnel / symétrisé sur hep-th")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-k", type=int, default=5000,
                        help="Maximum de prédictions représentées dans les courbes (défaut 5000)")
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.tol <= 0 or args.max_iter <= 0 or args.block_size <= 0:
        parser.error("alpha dans ]0,1[, tol > 0, max-iter et block-size > 0")

    print("Chargement des données hep-th...", flush=True)
    papers = load_hepth(DATA / "hep-th-abs.tar.gz", DATA / "hep-th-slacdates.tar.gz")
    split = temporal_split(papers, train=(1994, 1996), test=(1997, 1999),
                           k_train=3, k_test=3)
    core = sorted(split.core)
    n = len(core)
    k = len(split.new_edges)
    baseline = k / split.num_candidates
    print(f"Cœur : {n:,} auteurs | Paires candidates : {split.num_candidates:,} | Nouvelles arêtes : {k:,}")
    print(f"Hasard : {100 * baseline:.6f} %")
    print("Calcul de la matrice RWR une seule fois pour les trois variantes...", flush=True)
    begin = time.perf_counter()
    R, nodes, index = rwr_matrix(split.train_adj, core,
                                 alpha=args.alpha, tol=args.tol,
                                 max_iter=args.max_iter, block_size=args.block_size)
    compute_time = time.perf_counter() - begin
    print(f"RWR commun aux 3 variantes : {compute_time:.3f} s", flush=True)

    # Paires canoniques (u < v), ordonnées alphabétiquement.
    core_rows = np.array([index[u] for u in core], dtype=np.int64)
    ii, jj = np.triu_indices(n, k=1)
    allowed = np.fromiter(
        (core[j] not in split.train_adj[core[i]] for i, j in zip(ii, jj)),
        dtype=bool, count=len(ii),
    )
    ii, jj = ii[allowed], jj[allowed]
    assert len(ii) == split.num_candidates, (len(ii), split.num_candidates)
    y = np.fromiter(((core[i], core[j]) in split.new_edges for i, j in zip(ii, jj)),
                    dtype=bool, count=len(ii))
    assert int(np.count_nonzero(y)) == k

    # R[row_of_target, col_of_source] = r_source(target).
    forward = R[core_rows[jj], ii]  # u -> v, u < v (direction conventionnelle)
    backward = R[core_rows[ii], jj]  # v -> u, ordre inverse
    symmetric = forward + backward

    methods = {
        "Original_u_vers_v": forward,
        "Original_v_vers_u": backward,
        "RWR_symetrise": symmetric,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary, top_rows, pr_rows = [], [], []
    rank_index = {}
    max_k = min(args.max_k, len(ii))
    plot_ks = sorted(set([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, k, 2000, max_k] +
                         list(range(50, max_k + 1, 25))))
    plot_ks = [t for t in plot_ks if 1 <= t <= max_k]
    graph = {}

    for name, scores in methods.items():
        rank_start = time.perf_counter()
        order = np.argsort(-scores, kind="stable")
        rank_time = time.perf_counter() - rank_start
        rank_index[name] = order[:k]
        sorted_scores = scores[order]
        ordered_true = y[order]
        observed_hits = int(np.count_nonzero(ordered_true[:k]))
        expected_hits, ties, places = expected_top_k(sorted_scores, ordered_true, k)
        positive_mask = scores > 0
        recoverable = int(np.count_nonzero(y & positive_mask))
        row = [name, observed_hits, observed_hits / k, expected_hits,
               expected_hits / k, (expected_hits / k) / baseline,
               int(np.count_nonzero(positive_mask)), recoverable,
               ties, places, compute_time, rank_time]
        summary.append(row)
        print(f"\n{name}:")
        print(f"  Top-{k} déterministe : {observed_hits} vraies collaborations ({observed_hits / k:.4%})")
        print(f"  Ex æquo aléatoires en espérance : {expected_hits:.2f} ({expected_hits / k:.4%})")
        print(f"  Facteur vs hasard (espérance) : ×{(expected_hits/k)/baseline:.2f}")
        print(f"  Liens positifs à score > 0 : {recoverable}/{k}")
        print(f"  Ex æquo au seuil : {ties} pour {places} places")

        for position, ind in enumerate(order[:k], start=1):
            top_rows.append((name, position, core[ii[ind]], core[jj[ind]],
                             f"{scores[ind]:.17g}", int(y[ind])))

        points = []
        for t in plot_ks:
            e_hits, _, _ = expected_top_k(sorted_scores, ordered_true, t)
            precision, recall = e_hits / t, e_hits / k
            pr_rows.append((name, t, f"{precision:.12g}", f"{recall:.12g}",
                            f"{e_hits:.12g}"))
            points.append((t, precision, recall))
        graph[name] = points

    # Similarité des top-k (pour évaluer à quel point les classements changent).
    overlaps = []
    names = list(methods)
    for a_index in range(len(names)):
        for b_index in range(a_index + 1, len(names)):
            a, b = names[a_index], names[b_index]
            intersect = len(set(rank_index[a]) & set(rank_index[b]))
            overlaps.append((a, b, intersect, f"{intersect/k:.6f}"))
            print(f"Chevauchement Top-{k} {a} / {b} : {intersect}/{k}")

    write_csv(OUTPUT / "resume_comparaison_rwr.csv",
              ["methode", "tp_deterministe", "precision_deterministe",
               "tp_esperance_exaequo", "precision_esperance_exaequo",
               "facteur_hasard_esperance", "paires_score_positif", "vrais_liens_score_positif",
               "nb_exaequo_seuil", "places_exaequo", "temps_calcul_RWR_commun_s",
               "temps_classement_s"], summary)
    write_csv(OUTPUT / f"top_{k}_trois_variantes_rwr.csv",
              ["methode", "rang", "u", "v", "score", "nouvelle_relation"], top_rows)
    write_csv(OUTPUT / "courbes_rwr_variantes.csv",
              ["methode", "k", "precision_esperee", "rappel_espere", "tp_esperes"], pr_rows)
    write_csv(OUTPUT / "chevauchement_top_k.csv",
              ["methode_1", "methode_2", "intersection_top_k", "intersection_sur_k"], overlaps)

    with (OUTPUT / "protocole_comparaison.md").open("w", encoding="utf-8") as file:
        file.write(f"""# Comparaison RWR directionnel / symétrisé

- Données : hep-th KDD Cup du dépôt, dates SLAC.
- Apprentissage 1994–1996, test 1997–1999, cœur 3 articles par période.
- {n} auteurs cœur, {split.num_candidates} candidats, {k} nouvelles relations à retrouver.
- RWR : alpha={args.alpha}, tol={args.tol}, max_iter={args.max_iter}, block_size={args.block_size}.
- **Une seule exécution de `rwr_matrix`** produit la même matrice de probabilités pour toutes les variantes.
- `Original_u_vers_v` : r_u(v), où u < v selon l'ordre lexical des identifiants d'auteurs.
- `Original_v_vers_u` : r_v(u), inverse lexical ; les deux sont des formes directionnelles de la définition de l'article.
- `RWR_symetrise` : r_u(v) + r_v(u), la formule du dépôt.
- Le choix lexical est arbitraire et n'est PAS une convention imposée par l'article.
  Ne pas décrire le score `Original_u_vers_v` comme une reproduction littérale complète
  de l'évaluation historique sur des paires non orientées.
- Toutes les variantes utilisent exactement les mêmes paires candidates et les mêmes liens de test.
- En cas d'ex æquo, le Top-k déterministe utilise l'ordre lexical ; la métrique
  en espérance correspond à un départage uniforme au hasard dans chaque groupe.
- Les paires de score nul restent dans les candidats. Les courbes ici s'arrêtent à k={max_k}.
- Les temps de calcul RWR ne doivent PAS être additionnés ni comparés entre variantes :
  la même matrice a été calculée une seule fois. Seuls les temps de tri diffèrent.
- Si les métriques changent selon l'ordre, cela révèle une sensibilité du RWR directionnel
  à la convention choisie pour les paires non orientées.
- Corpus différent de celui de l'article original, comparaisons internes seulement.
""")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, points in graph.items():
            ax.plot([p[2] * 100 for p in points], [p[1] * 100 for p in points],
                    label=name)
        ax.axhline(baseline * 100, linestyle="--", linewidth=1.3,
                   label="Hasard (espérance)")
        ax.set_xlabel("Rappel attendu (%)")
        ax.set_ylabel("Précision attendue (%)")
        ax.set_title("hep-th — RWR directionnel vs symétrisé (départage aléatoire en espérance)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / "precision_rappel_rwr_comparaison.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("Matplotlib non installé : graphiques non générés (CSV disponibles).")

    print(f"\nTerminé. Résultats : {OUTPUT.relative_to(ROOT)}")



if __name__ == "__main__":
    main()
