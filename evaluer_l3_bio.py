"""Évalue CN, AA et L3 sur le même train/test HI-II-14.

Usage (racine du dépôt):
    python evaluer_l3_bio.py --data results/l3_biologique --output results/l3_biologique/evaluation

Requis : src.measures.l3.l3_optimise, src.measures.local.cn_scores/aa_scores

Les paires à score nul sont prises en compte analytiquement (ex æquo uniformes
en espérance). On n'énumère jamais les ~8,7 millions de candidats.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from heapq import nsmallest
from pathlib import Path

from src.graph_io import from_edges
from src.measures.local import aa_scores, cn_scores
from src.measures.l3 import l3_optimise


def lire_aretes(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    pairs: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_num}: 2 colonnes attendues")
            u, v = parts[0].strip(), parts[1].strip()
            if not u or not v or u == v:
                raise ValueError(f"{path}:{line_num}: arête invalide {u!r} {v!r}")
            pairs.add((u, v) if u < v else (v, u))
    return pairs


def courbe_attendue(scores, vrais_liens, total_candidats, max_k):
    """Courbe de précision/rappel en espérance avec tirage uniforme des ex æquo.

    Une paire de score absent est de score zéro. O(P+G log G+max_k)
    plutôt que O(nombre_total_de_candidats).
    """
    if max_k <= 0 or total_candidats <= 0 or not vrais_liens:
        raise ValueError("Effectifs invalides")
    if len(scores) > total_candidats:
        raise ValueError("Davantage de scores que de candidats")

    groupes = defaultdict(lambda: [0, 0])
    hits_positifs = 0
    for pair, score in scores.items():
        if not math.isfinite(score) or score <= 0:
            raise ValueError(f"Score non positif ou non fini : {pair}: {score}")
        hit = pair in vrais_liens
        groupes[score][0] += 1
        groupes[score][1] += int(hit)
        hits_positifs += hit

    nb_zero = total_candidats - len(scores)
    hits_zero = len(vrais_liens) - hits_positifs
    if not 0 <= hits_zero <= nb_zero:
        raise ValueError("Incohérence dans les liens de test et les scores")
    if nb_zero:
        groupes[0.0] = [nb_zero, hits_zero]

    k_cap = min(max_k, total_candidats)
    rows = []
    k = 0
    hits_cumules = 0.0
    for score in sorted(groupes, reverse=True):
        nombre, bons = groupes[score]
        for j in range(1, min(nombre, k_cap - k) + 1):
            kk = k + j
            hits_moyens = hits_cumules + j * bons / nombre
            rows.append((kk, hits_moyens / kk, hits_moyens / len(vrais_liens)))
        k += nombre
        hits_cumules += bons
        if k >= k_cap:
            break
    return rows, hits_positifs


def top_k_deterministe(scores, vrais_liens, k, n_candidats):
    """Sans trier tous les scores. Complétion des scores nuls non matérialisée.

    Ce résultat est strictement déterministe seulement si tous les k meilleurs
    candidats ont un score > 0. Sinon, on renvoie None (on retient l'espérance).
    """
    if len(scores) < k:
        return None
    choix = nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))
    return sum(pair in vrais_liens for pair, _ in choix)


def ecrire_csv(path, rows, noms):
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(noms)
        wr.writerows(rows)


def graphes(rows_by_method, base_aleatoire, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : CSV générés mais pas les graphiques")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in rows_by_method.items():
        xs = [x[2] * 100 for x in rows]
        ys = [x[1] * 100 for x in rows]
        ax.plot(xs, ys, label=name, linewidth=1.7)
    ax.axhline(base_aleatoire * 100, linestyle="--", color="gray", label="Hasard (espérance)")
    ax.set_xlabel("Rappel (%)")
    ax.set_ylabel("Précision (%)")
    ax.set_title("HI-II-14 — précision-rappel, ex æquo uniformes en espérance")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_rappel_l3_cn_aa.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in rows_by_method.items():
        ax.plot([x[0] for x in rows], [x[1] * 100 for x in rows], label=name, linewidth=1.7)
    ax.axhline(base_aleatoire * 100, linestyle="--", color="gray", label="Hasard (espérance)")
    ax.set_xlabel("Nombre de prédictions k")
    ax.set_ylabel("Précision@k (%)")
    ax.set_title("HI-II-14 — précision selon le nombre de prédictions")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_selon_k_l3_cn_aa.png", dpi=170)
    plt.close(fig)


def evaluer(data, output, max_k):
    train = lire_aretes(data / "train_edges.tsv")
    test = lire_aretes(data / "test_edges.tsv")
    if not train or not test:
        raise ValueError("Ensembles train/test vides")
    if train & test:
        raise ValueError("Fuite de données : train et test se chevauchent")
    # Les IDs de protéines sont connus (protocole transductif), y compris
    # ceux qui n'ont plus aucune arête après suppression des boucles.
    # Aucune arête de test n'est ajoutée au graphe d'apprentissage.
    nodes_path = data / 'nodes.tsv'
    if not nodes_path.is_file():
        raise FileNotFoundError(
            f'{nodes_path} absent : relance preparer_donnees_l3.py '
            'pour conserver les sommets isolés.'
        )
    with nodes_path.open(encoding='utf-8-sig') as fh:
        nodes = {s for line in fh if (s := line.strip()) and not s.startswith('#')}
    ends = {node for uv in train | test for node in uv}
    if not ends <= nodes:
        raise ValueError(f'{len(ends - nodes)} sommet(s) train/test absent(s) de nodes.tsv')
    protocole_source = data / 'protocole.json'
    if protocole_source.is_file():
        original = json.loads(protocole_source.read_text(encoding='utf-8'))
        if len(nodes) != original.get('sommets_total'):
            raise ValueError('Incohérence du nombre de sommets avec protocole.json')
    adj = from_edges(train, nodes=nodes)
    nb_candidates = len(nodes) * (len(nodes) - 1) // 2 - len(train)
    if protocole_source.is_file() and nb_candidates != original.get('paires_candidates'):
        raise ValueError('Incohérence du nombre de candidats avec protocole.json')
    baseline = len(test) / nb_candidates
    isolates = {u for u in adj if not adj[u]}
    test_inaccessibles = sum(u in isolates or v in isolates for u, v in test)
    k = len(test)
    plot_k = min(max_k or max(2 * k, 20000), nb_candidates)

    output.mkdir(parents=True, exist_ok=True)
    print("=== Évaluation biologique CN / AA / L3 ===", flush=True)
    print(f"Sommets : {len(nodes):,} ; train : {len(train):,} ; test : {len(test):,}", flush=True)
    print(f"Candidats : {nb_candidates:,} ; hasard : {100 * baseline:.6f} %", flush=True)
    print(f"Isolés dans train : {len(isolates):,} ; arêtes test impliquant un isolé : {test_inaccessibles:,}/{len(test):,}", flush=True)
    print(f"Évaluation jusqu'à k={plot_k:,} ; seuil principal k={k:,}", flush=True)

    methods = {"CN": cn_scores, "AA": aa_scores, "L3_optimise": l3_optimise}
    summaries = []
    curves = {}
    for name, func in methods.items():
        print(f"\nCalcul {name}...", flush=True)
        debut = time.perf_counter()
        scores = func(adj)
        duree = time.perf_counter() - debut
        if any(u == v or v in adj[u] for u, v in scores):
            raise AssertionError(f"{name}: une arête existante ou une boucle est prédite")
        rows, hits_pos = courbe_attendue(scores, test, nb_candidates, plot_k)
        hits_det = top_k_deterministe(scores, test, k, nb_candidates)
        krow = rows[k - 1]
        hits_exp = krow[1] * k
        summaries.append((name, len(scores), hits_pos, f"{duree:.6f}",
                          "" if hits_det is None else hits_det, f"{hits_exp:.6f}",
                          f"{100 * krow[1]:.6f}", f"{100 * krow[2]:.6f}",
                          f"{krow[1] / baseline:.4f}"))
        curves[name] = rows
        print(f"  Scores > 0 : {len(scores):,} ; temps calcul : {duree:.3f} s", flush=True)
        print(f"  Liens de test avec score > 0 : {hits_pos:,}/{len(test):,}", flush=True)
        print(f"  Top-{k}: {hits_exp:.2f} vrais liens (espérance), précision {100 * krow[1]:.4f} %, facteur hasard x{krow[1]/baseline:.2f}", flush=True)
        if hits_det is not None:
            print(f"  Top-{k} déterministe : {hits_det} vrais liens", flush=True)
        else:
            print("  Top-k déterministe : omis (score nul parmi les prédictions)", flush=True)
        del scores

    ecrire_csv(output / "resume_l3_bio.csv", summaries,
               ["methode", "nb_scores_positifs", "liens_test_score_positif", "temps_calcul_s",
                "vrais_top_k_deterministe", "vrais_top_k_esperance", "precision_k_pourcent",
                "rappel_k_pourcent", "facteur_vs_hasard"])
    ecrire_csv(output / "courbes_l3_cn_aa.csv",
               ((method, kk, f"{p:.12g}", f"{r:.12g}")
                for method, rows in curves.items() for kk, p, r in rows),
               ["methode", "k", "precision_attendue", "rappel_attendu"])
    graphes(curves, baseline, output)
    protocole = {
        "fraction_test": len(test) / (len(train) + len(test)),
        "sommets": len(nodes), "train": len(train), "test": len(test),
        "candidats": nb_candidates, "hasard_precision": baseline,
        "sommets_isoles_train": len(isolates),
        "liens_test_avec_au_moins_un_isole": test_inaccessibles,
        "k_principal": k, "k_max_courbes": plot_k,
        "tie_policy": "uniforme aléatoire en espérance, y compris score 0",
        "note": "L'absence d'interaction dans la base n'est pas une preuve de non-interaction biologique."
    }
    (output / "protocole_evaluation.json").write_text(
        json.dumps(protocole, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] Résultats dans : {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation HI-II-14 de CN, AA et L3")
    parser.add_argument("--data", type=Path, default=Path("results/l3_biologique"))
    parser.add_argument("--out", type=Path, default=Path("results/l3_biologique/evaluation"))
    parser.add_argument("--max-k", type=int, default=None,
                        help="Longueur des courbes ; défaut max(2*k, 20000)")
    args = parser.parse_args()
    evaluer(args.data, args.out, args.max_k)
