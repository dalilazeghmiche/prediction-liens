"""Évaluation CN / AA / L3 avec ex æquo uniformes, y compris scores nuls."""
from __future__ import annotations

import argparse
import json
import time
from heapq import nsmallest
from pathlib import Path

from src.graph_io import from_edges
from utils_experiences import courbe_attendue, save_csv, title, result, done
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



def top_k_deterministe(scores, vrais_liens, k, n_candidats):
    """Sans trier tous les scores. Complétion des scores nuls non matérialisée.

    Ce résultat est strictement déterministe seulement si tous les k meilleurs
    candidats ont un score > 0. Sinon, on renvoie None (on retient l'espérance).
    """
    if len(scores) < k:
        return None
    choix = nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))
    return sum(pair in vrais_liens for pair, _ in choix)



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
    title("HI-II-14 | CN / AA / L3", info=f"{len(nodes):,} protéines | {nb_candidates:,} candidats | k={k}")

    methods = {"CN": cn_scores, "AA": aa_scores, "L3_optimise": l3_optimise}
    summaries = []
    curves = {}
    for name, func in methods.items():
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
        result(name, f"P@{k} = {100 * krow[1]:.2f} %",
               f"{hits_exp:.1f} liens attendus", seconds=duree)
        del scores

    save_csv(output / "resume_l3_bio.csv", summaries,
               ["methode", "nb_scores_positifs", "liens_test_score_positif", "temps_calcul_s",
                "vrais_top_k_deterministe", "vrais_top_k_esperance", "precision_k_pourcent",
                "rappel_k_pourcent", "facteur_vs_hasard"])
    save_csv(output / "courbes_l3_cn_aa.csv",
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
    done(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation HI-II-14 de CN, AA et L3")
    parser.add_argument("--data", type=Path, default=Path("results/l3_biologique"))
    parser.add_argument("--out", type=Path, default=Path("results/l3_biologique/evaluation"))
    parser.add_argument("--max-k", type=int, default=None,
                        help="Longueur des courbes ; défaut max(2*k, 20000)")
    args = parser.parse_args()
    evaluer(args.data, args.out, args.max_k)
