"""Teste la robustesse de CN, AA, L3 sur plusieurs graines de masquage."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from utils_experiences import title, result, done

from preparer_donnees_l3 import lire_hi_ii_14
from evaluer_l3_bio import courbe_attendue
from src.split import random_edge_split
from src.measures.local import cn_scores, aa_scores
from src.measures.l3 import l3_optimise

METHODES = {
    "CN": cn_scores,
    "AA": aa_scores,
    "L3_optimise": l3_optimise,
}


def parser_graines(texte: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in texte.split(",")]
    except ValueError as e:
        raise ValueError("--seeds doit contenir des entiers separes par des virgules") from e
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Indiquer au moins deux graines distinctes")
    return seeds


def ecrire_csv(path: Path, lignes: list[dict], champs: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=champs)
        writer.writeheader()
        writer.writerows(lignes)


def resume_par_methode(lignes: list[dict]) -> list[dict]:
    resultat = []
    for methode in METHODES:
        selection = [row for row in lignes if row["methode"] == methode]
        precisions = [row["precision_k_pourcent"] for row in selection]
        rappels_pos = [row["rappel_positif_pourcent"] for row in selection]
        temps = [row["temps_scores_s"] for row in selection]
        resultat.append({
            "methode": methode,
            "nb_graines": len(selection),
            "precision_moyenne_pct": statistics.mean(precisions),
            "precision_ecart_type_pct": statistics.stdev(precisions),
            "precision_min_pct": min(precisions),
            "precision_max_pct": max(precisions),
            "rappel_positif_moyen_pct": statistics.mean(rappels_pos),
            "temps_scores_moyen_s": statistics.mean(temps),
        })
    return resultat


def deltas_apparies(lignes: list[dict]) -> list[dict]:
    par_graine = defaultdict(dict)
    for row in lignes:
        par_graine[row["seed"]][row["methode"]] = row
    resultats = []
    for seed, d in sorted(par_graine.items()):
        for autre in ("CN", "AA"):
            resultats.append({
                "seed": seed,
                "comparaison": f"L3_optimise - {autre}",
                "gain_points_precision": (
                    d["L3_optimise"]["precision_k_pourcent"]
                    - d[autre]["precision_k_pourcent"]
                ),
                "ratio_precisions": (
                    d["L3_optimise"]["precision_k_pourcent"]
                    / d[autre]["precision_k_pourcent"]
                    if d[autre]["precision_k_pourcent"] > 0 else None
                ),
            })
    return resultats


def tracer(lignes: list[dict], resume: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : CSV ecrits, graphiques ignores", flush=True)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for methode in METHODES:
        part = sorted((r for r in lignes if r["methode"] == methode),
                      key=lambda r: r["seed"])
        ax.plot([r["seed"] for r in part],
                [r["precision_k_pourcent"] for r in part],
                marker="o", label=methode)
    ax.set(xlabel="Graine du decoupage", ylabel="Precision@|test| (%)",
           title="HI-II-14 : robustesse de la precision selon la graine")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "precision_par_graine.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    noms = [r["methode"] for r in resume]
    moy = [r["precision_moyenne_pct"] for r in resume]
    sd = [r["precision_ecart_type_pct"] for r in resume]
    ax.bar(noms, moy, yerr=sd, capsize=5)
    ax.set(ylabel="Precision@|test| moyenne (%)",
           title="HI-II-14 : moyenne +/- ecart-type (5 decoupages)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "precision_moyenne_ecart_type.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for methode in METHODES:
        part = sorted((r for r in lignes if r["methode"] == methode),
                      key=lambda r: r["seed"])
        ax.plot([r["seed"] for r in part],
                [r["rappel_positif_pourcent"] for r in part],
                marker="o", label=methode)
    ax.set(xlabel="Graine du decoupage", ylabel="Rappel atteignable avec score > 0 (%)",
           title="HI-II-14 : couverture des interactions masquees")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "couverture_par_graine.png", dpi=170)
    plt.close(fig)


def executer(source: Path, out: Path, seeds: list[int], fraction: float,
             figures: bool = True) -> tuple[list[dict], list[dict]]:
    if not 0 < fraction < 1:
        raise ValueError("La fraction de test doit etre dans ]0,1[")
    graphe, info = lire_hi_ii_14(source)
    n = len(graphe)
    m = info["aretes_uniques"]
    title("HI-II-14 | Robustesse", info=f"{n:,} protéines | {len(seeds)} graines")

    out.mkdir(parents=True, exist_ok=True)
    lignes = []
    par_seed = []
    for seed in seeds:
        split = random_edge_split(graphe, test_fraction=fraction, seed=seed)
        train = split.train_adj
        test = split.new_edges
        k = len(test)
        candidats = split.num_candidates
        if k <= 0 or candidats <= 0:
            raise ValueError("Decoupage vide : augmenter la taille des donnees ou la fraction")
        assert k + sum(len(nbrs) for nbrs in train.values()) // 2 == m
        assert all(v not in train[u] for u, v in test)
        assert len(train) == n
        hasard = 100 * k / candidats
        isoles = {u for u in train if not train[u]}
        liens_isoles = sum(u in isoles or v in isoles for u, v in test)
        recap = []
        par_seed.append({"seed": seed, "n": n, "train": m - k,
                         "test": k, "candidats": candidats,
                         "hasard_precision_pourcent": hasard,
                         "sommets_isoles": len(isoles),
                         "liens_test_avec_isole": liens_isoles})
        for methode, algo in METHODES.items():
            debut = time.perf_counter()
            scores = algo(train)
            duree = time.perf_counter() - debut
            # Evaluations en esperance avec gestion exacte des ex aequo et des zeros
            courbe, nb_hits_positifs = courbe_attendue(scores, test, candidats, k)
            _, precision, rappel = courbe[-1]
            if any(u == v or v in train[u] for u, v in scores):
                raise AssertionError(f"{methode} : paire interdite predite")
            row = {
                "seed": seed, "methode": methode,
                "n_sommets": n, "n_train": m - k, "n_test": k,
                "n_candidats": candidats,
                "hasard_precision_pourcent": hasard,
                "nb_scores_positifs": len(scores),
                "liens_test_score_positif": nb_hits_positifs,
                "rappel_positif_pourcent": 100 * nb_hits_positifs / k,
                "vrais_top_k_esperance": precision * k,
                "precision_k_pourcent": 100 * precision,
                "rappel_k_pourcent": 100 * rappel,
                "facteur_vs_hasard": 100 * precision / hasard,
                "temps_scores_s": duree,
            }
            lignes.append(row)
            recap.append(f"{methode}={row['precision_k_pourcent']:.2f}%")
            del scores, courbe
            gc.collect()
        del train, split
        print(f"  seed {seed}: " + " | ".join(recap), flush=True)

    resume = resume_par_methode(lignes)
    deltas = deltas_apparies(lignes)
    ecrire_csv(out / "mesures_par_graine.csv", lignes, list(lignes[0]))
    ecrire_csv(out / "description_decoupages.csv", par_seed, list(par_seed[0]))
    ecrire_csv(out / "resume_robustesse.csv", resume, list(resume[0]))
    ecrire_csv(out / "comparaisons_appariees.csv", deltas, list(deltas[0]))
    if figures:
        tracer(lignes, resume, out)
    (out / "protocole_robustesse.json").write_text(json.dumps({
        "source": str(source),
        "nb_sommets": n, "nb_aretes_apres_nettoyage": m,
        "graines": seeds, "fraction_test": fraction,
        "methodes": list(METHODES), "k": "nombre d'aretes masquees pour la graine",
        "tie_policy": "departage uniforme aleatoire en esperance, y compris score 0",
        "sommets": "tous conserves, y compris isoles",
        "fichiers_seed_42": "aucun ecrasement : tous les decoupages sont en memoire",
        "precision": "vrais liens attendus au Top-k / k",
        "ecart_type": "ecart-type empirique echantillon (ddof=1)",
        "limites": ["5 graines ne constituent pas une validation exhaustive",
                     "une seule source biologique",
                     "les liens absents peuvent etre des interactions inconnues",
                     "le masquage peut isoler certaines proteines"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("  Moyenne ± écart-type (points de %) :")
    for r in resume:
        print(f"    {r['methode']:<12} {r['precision_moyenne_pct']:.2f}% ± {r['precision_ecart_type_pct']:.2f}")
    done(out)
    return lignes, resume


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustesse CN / AA / L3 - HI-II-14")
    parser.add_argument("--source", type=Path, default=Path("data/raw/HI-II-14.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/l3_biologique/robustesse"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--no-plots", action="store_true", help="Desactiver les figures")
    args = parser.parse_args()
    executer(args.source, args.out, parser_graines(args.seeds),
             args.fraction, figures=not args.no_plots)


if __name__ == "__main__":
    main()
