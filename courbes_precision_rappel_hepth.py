"""Courbes précision-rappel CN, Adamic-Adar et RWR sur hep-th.


    .\\.venv\\Scripts\\python.exe -m pip install matplotlib
    .\\.venv\\Scripts\\python.exe courbes_precision_rappel_hepth.py

Protocole : apprentissage 1994-1996 ; test 1997-1999 ; cœur 3 articles
minimum dans chacune des deux périodes. Comparaison sur l'intégralité des
paires du cœur non reliées pendant l'apprentissage (y compris score zéro).

Chaque méthode construit les scores existants, puis le script regroupe les
paires ex æquo. Pour un nombre k de liens sélectionnés, le script calcule
l'espérance du nombre de vrais positifs si les ex æquo sont départagés
uniformément au hasard. Le calcul est exact, sans simulation aléatoire.

Les scores nuls non renvoyés par les fonctions des algorithmes sont regroupés
sans construire explicitement leurs ~centaines de milliers de paires.

Sorties dans results/courbes_precision_rappel_hepth/ :
    courbe_precision_rappel_complete.png
    courbe_precision_rappel_zoom.png
    courbes_precision_rappel.csv
    synthese_precision_rappel.csv
    protocole_et_limites.txt
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
from pathlib import Path
import time

# Important : backend sans interface graphique pour la génération de PNG.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import load_hepth
from src.measures.local import cn_scores, aa_scores, rank
from src.measures.global_measures import rwr_scores
from src.split import temporal_split
from src.evaluate import random_precision


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "courbes_precision_rappel_hepth"


@dataclass
class RankingStats:
    name: str
    num_candidates: int
    num_positives: int
    positive_scores: int
    positives_positive_scores: int
    elapsed: float
    ends: list[int]
    cumulative_hits: list[int]
    group_sizes: list[int]
    group_hits: list[int]
    deterministic_hits_at_k: int
    tie_size_at_k: int
    tie_slots_at_k: int

    def expected_hits(self, k: int) -> float:
        """Espérance exacte des TP au top-k, ex æquo départagés au hasard."""
        if not (1 <= k <= self.num_candidates):
            raise ValueError("k doit être entre 1 et le nombre de candidats")
        i = bisect_left(self.ends, k)
        previous_end = 0 if i == 0 else self.ends[i - 1]
        previous_hits = 0 if i == 0 else self.cumulative_hits[i - 1]
        return previous_hits + (k - previous_end) * self.group_hits[i] / self.group_sizes[i]

    def expected_precision_recall(self, k: int) -> tuple[float, float]:
        hits = self.expected_hits(k)
        return hits / k, hits / self.num_positives


def analyse_ranking(name, scores, positives, num_candidates, k_reference, started_at):
    """Traite seulement les scores > 0 et représente le reste par un bloc nul."""
    if len(scores) > num_candidates:
        raise AssertionError("Plus de scores que de paires candidates")
    if any(score <= 0 for score in scores.values()):
        raise AssertionError("Les scores retournés doivent être strictement positifs")

    ranked = rank(scores)  # même règle déterministe que le projet
    deterministic_hits = sum(pair in positives for pair, _ in ranked[:k_reference])

    ends, cumulative_hits, group_sizes, group_hits = [], [], [], []
    last_score = None
    group_size, group_pos = 0, 0
    total_count, total_pos = 0, 0

    def emit_group():
        nonlocal total_count, total_pos
        if group_size:
            total_count += group_size
            total_pos += group_pos
            ends.append(total_count)
            cumulative_hits.append(total_pos)
            group_sizes.append(group_size)
            group_hits.append(group_pos)

    for pair, score in ranked:
        if last_score is not None and score != last_score:
            emit_group()
            group_size, group_pos = 0, 0
        last_score = score
        group_size += 1
        group_pos += int(pair in positives)
    emit_group()

    if total_count != len(scores):
        raise AssertionError("Erreur interne dans les groupes ex æquo")

    # Compléter avec toutes les paires non produites par l'algorithme (score 0).
    zeros = num_candidates - len(scores)
    positives_in_zeros = len(positives) - total_pos
    if not 0 <= positives_in_zeros <= zeros:
        raise AssertionError("Les positifs dans le groupe zéro sont incohérents")
    if zeros:
        ends.append(num_candidates)
        cumulative_hits.append(len(positives))
        group_sizes.append(zeros)
        group_hits.append(positives_in_zeros)

    if ends[-1] != num_candidates or cumulative_hits[-1] != len(positives):
        raise AssertionError("Tous les candidats et vrais liens doivent être représentés")

    idx = bisect_left(ends, k_reference)
    above = ends[idx - 1] if idx else 0
    stats = RankingStats(
        name=name,
        num_candidates=num_candidates,
        num_positives=len(positives),
        positive_scores=len(scores),
        positives_positive_scores=total_pos,
        elapsed=time.perf_counter() - started_at,
        ends=ends,
        cumulative_hits=cumulative_hits,
        group_sizes=group_sizes,
        group_hits=group_hits,
        deterministic_hits_at_k=deterministic_hits,
        tie_size_at_k=group_sizes[idx],
        tie_slots_at_k=k_reference - above,
    )
    return stats


def selection_ks(total, k_ref):
    """Densité fine au début ; échantillonnage progressif vers tous les candidats."""
    ks = set(range(1, min(2500, total) + 1))
    ks.update(range(2501, min(total, 10000) + 1, 10))
    ks.update(range(10001, min(total, 100000) + 1, 250))
    ks.update(range(100001, total + 1, 2000))
    ks.update({total, k_ref, min(100, total), min(500, total), min(1000, total)})
    return sorted(k for k in ks if 1 <= k <= total)


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
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Probabilité de redémarrage RWR (défaut 0.15)")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="Tolérance de convergence RWR (défaut 1e-10)")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Itérations RWR maximales (défaut 1000)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Taille des blocs RWR (défaut 64)")
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.tol <= 0 or args.max_iter <= 0 or args.block_size <= 0:
        parser.error("alpha dans ]0,1[, tol > 0, max-iter et block-size > 0")

    files = (RAW / "hep-th-abs.tar.gz", RAW / "hep-th-slacdates.tar.gz")
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"Archive introuvable : {f}")

    print("Chargement hep-th et découpage temporel...", flush=True)
    papers = load_hepth(*files)
    split = temporal_split(papers, train=(1994, 1996), test=(1997, 1999),
                           k_train=3, k_test=3)
    k = len(split.new_edges)
    n = split.num_candidates
    random_baseline = random_precision(k, n)
    core_sorted = sorted(split.core)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Cœur : {len(split.core):,} auteurs | Candidats : {n:,} | Nouveaux liens : {k:,}")
    print(f"Précision au hasard attendue : {random_baseline:.6%}\n")

    experiments = [
        ("CN", lambda: cn_scores(split.train_adj, nodes=split.core)),
        ("AA", lambda: aa_scores(split.train_adj, nodes=split.core)),
        ("RWR_symetrise", lambda: rwr_scores(
            split.train_adj, nodes=core_sorted, alpha=args.alpha, tol=args.tol,
            max_iter=args.max_iter, block_size=args.block_size)),
    ]
    stats = []
    for name, run in experiments:
        print(f"Calcul {name}...", flush=True)
        started = time.perf_counter()
        scores = run()
        s = analyse_ranking(name, scores, split.new_edges, n, k, started)
        stats.append(s)
        precision, recall = s.expected_precision_recall(k)
        print(f"  scores > 0 : {s.positive_scores:,} | temps calcul + tri : {s.elapsed:.3f} s")
        print(f"  top-{k} déterministe : {s.deterministic_hits_at_k} bonnes prédictions")
        print(f"  top-{k} ex æquo aléatoires (espérance) : {s.expected_hits(k):.2f} bonnes prédictions")
        print(f"  précision et rappel attendus @ {k} : {precision:.4%} / {recall:.4%}")
        print(f"  positifs récupérables avec score > 0 : {s.positives_positive_scores}/{k}\n", flush=True)
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
    print("Fichiers générés :")
    for f in (csv_path, summary_path, image1, image2, note):
        print(" -", f.relative_to(ROOT))



if __name__ == "__main__":
    main()
