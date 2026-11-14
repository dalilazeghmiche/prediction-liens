"""Courbes précision-rappel CN, AA et RWR sur hep-th (ex æquo et scores nuls inclus)."""

from __future__ import annotations

import argparse
import csv
from utils_experiences import (RESULTS, social_split, title, result, done,
                               rwr_options, check_rwr, analyse_ranking, selection_ks)
import time

# Important : backend sans interface graphique pour la génération de PNG.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.measures.local import cn_scores, aa_scores
from src.measures.global_measures import rwr_scores


OUT = RESULTS / "courbes_precision_rappel_hepth"



def save_plot(stats, ks, random_baseline, k_reference, path, zoom):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in stats:
        x, y = [], []
        for k in ks:
            p, r = s.expected_precision_recall(k)
            x.append(100 * r)
            y.append(100 * p)
        ax.plot(x, y, linewidth=1.7, label=s.name)
        pr, re = s.expected_precision_recall(k_reference)
        ax.scatter([100 * re], [100 * pr], s=26, zorder=4)
    ax.axhline(100 * random_baseline, linestyle="--", linewidth=1,
               color="gray", label="Hasard (espérance)")
    ax.set(xlabel="Rappel (%)", ylabel="Précision (%)",
           title="hep-th — courbes précision-rappel (ex æquo aléatoires, en espérance)")
    if zoom:
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 30)
        ax.set_title("hep-th — précision-rappel, zoom (rappel ≤ 25 %, précision ≤ 30 %)")
    else:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Courbes précision-rappel : CN, AA, RWR sur hep-th")
    rwr_options(parser)
    args = parser.parse_args()
    check_rwr(parser, args)

    split = social_split()
    k, n = len(split.new_edges), split.num_candidates
    random_baseline = k / n
    core_sorted = sorted(split.core)
    OUT.mkdir(parents=True, exist_ok=True)
    title("hep-th | Courbes P-R", info=f"{len(split.core):,} auteurs | {n:,} candidats | k={k}")

    experiments = [
        ("CN", lambda: cn_scores(split.train_adj, nodes=split.core)),
        ("AA", lambda: aa_scores(split.train_adj, nodes=split.core)),
        ("RWR_symetrise", lambda: rwr_scores(
            split.train_adj, nodes=core_sorted, alpha=args.alpha, tol=args.tol,
            max_iter=args.max_iter, block_size=args.block_size)),
    ]
    stats = []
    for name, run in experiments:
        started = time.perf_counter()
        scores = run()
        s = analyse_ranking(name, scores, split.new_edges, n, k, started)
        stats.append(s)
        precision, recall = s.expected_precision_recall(k)
        result(name, f"P@{k} = {precision:.2%}",
               f"{s.expected_hits(k):.1f} liens attendus", seconds=s.elapsed)
        del scores

    ks = selection_ks(n, k)
    csv_path = OUT / "courbes_precision_rappel.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithme", "k", "tp_attendus_departage_aleatoire",
                         "precision_attendue", "rappel_attendu", "precision_hasard"])
        for s in stats:
            for point in ks:
                precision, recall = s.expected_precision_recall(point)
                writer.writerow([s.name, point, f"{s.expected_hits(point):.10f}",
                                 f"{precision:.12f}", f"{recall:.12f}",
                                 f"{random_baseline:.12f}"])

    summary_path = OUT / "synthese_precision_rappel.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithme", "candidats", "nouveaux_liens", "k_reference",
                         "scores_positifs", "liens_reels_score_positif", "tp_topk_deterministe",
                         "tp_topk_exaequo_moyen", "precision_topk_deterministe",
                         "precision_topk_moyenne", "rappel_topk_moyen", "exaequo_groupe_seuil",
                         "exaequo_places", "secondes_calcul_tri_analyse"])
        for s in stats:
            precision, recall = s.expected_precision_recall(k)
            writer.writerow([s.name, n, k, k, s.positive_scores,
                             s.positives_positive_scores, s.deterministic_hits_at_k,
                             f"{s.expected_hits(k):.8f}",
                             f"{s.deterministic_hits_at_k/k:.12f}",
                             f"{precision:.12f}", f"{recall:.12f}",
                             s.tie_size_at_k, s.tie_slots_at_k,
                             f"{s.elapsed:.6f}"])

    image1 = OUT / "courbe_precision_rappel_complete.png"
    image2 = OUT / "courbe_precision_rappel_zoom.png"
    save_plot(stats, ks, random_baseline, k, image1, zoom=False)
    save_plot(stats, ks, random_baseline, k, image2, zoom=True)

    note = OUT / "protocole_et_limites.txt"
    note.write_text(
        "PROTOCOLE\n"
        "- Corpus : hep-th KDD Cup, lecture du ZIP du projet, années SLAC.\n"
        "- Apprentissage : 1994-1996 ; test : 1997-1999.\n"
        "- Cœur : auteurs avec au moins 3 articles dans chaque période.\n"
        f"- Candidats : {n:,} ; arêtes nouvelles : {k:,}.\n"
        "- RWR : score symétrisé r_u(v)+r_v(u) (adaptation de l'article [2]).\n"
        "- Scores > 0 : classés selon le score décroissant.\n"
        "- Paires non renvoyées : traitées comme groupe ex aequo de score zéro.\n"
        "- Pour chaque k : moyenne analytique de TP sur les départages aléatoires\n"
        "  à score identique ; pas de simulation.\n"
        "- La précision et le rappel attendus sont tracés en fonction de k.\n"
        "- Le zoom ne montre que rappel <= 25 % et précision <= 30 %.\n"
        "- Les points de la courbe sont échantillonnés en k ; le CSV contient\n"
        "  les valeurs exactes attendues aux k enregistrés.\n"
        "- Les résultats dépendent du corpus et du paramétrage RWR.\n"
        "- Le corpus du projet diffère du corpus original de l'article ;\n"
        "  la reproduction n'est pas numériquement identique.\n"
        "- Précision/rappel de cette expérience : mesurés hors des données\n"
        "  utilisées pour calculer les scores (hormis le critère historique\n"
        "  d'activité du cœur sur les deux périodes).\n",
        encoding="utf-8",
    )
    done(OUT)


if __name__ == "__main__":
    main()
