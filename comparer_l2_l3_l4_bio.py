"""Compare les marches A²/A³/A⁴ et la méthode L3 normalisée."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from utils_experiences import title, result, done
import math
import time

import numpy as np
from scipy.sparse import csr_matrix

from src.graph_io import from_edges
from src.measures.l3 import l3_optimise


def lire_aretes(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Fichier manquant: {path}")
    edges = set()
    for num, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            raise ValueError(f"{path}:{num}: 2 colonnes attendues")
        x, y = parts[0].strip(), parts[1].strip()
        if not x or not y or x == y:
            raise ValueError(f"{path}:{num}: arête invalide")
        edges.add((x, y) if x < y else (y, x))
    return edges


def lire_sommets(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Fichier manquant: {path}. Relancer preparer_donnees_l3.py")
    nodes = {line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines()
             if line.strip() and not line.startswith('#')}
    return sorted(nodes)


def creer_matrice(nodes, train):
    n = len(nodes)
    ids = {node: i for i, node in enumerate(nodes)}
    rows, cols = [], []
    voisins_train = [set() for _ in nodes]
    for x, y in train:
        i, j = ids[x], ids[y]
        rows.extend((i, j))
        cols.extend((j, i))
        voisins_train[i].add(j)
        voisins_train[j].add(i)
    a = csr_matrix((np.ones(len(rows), dtype=np.int64),
                    (np.array(rows), np.array(cols))), shape=(n, n), dtype=np.int64)
    a.sum_duplicates()
    a.sort_indices()
    return a, ids, voisins_train


def histogramme_scores(matrice, voisins_train, voisins_test):
    """Compte groupes (score -> [nb candidats, nb test]) sans créer de dict de paires."""
    h = defaultdict(lambda: [0, 0])
    n_positifs = 0
    vrais_positifs = 0
    for i in range(matrice.shape[0]):
        debut, fin = matrice.indptr[i], matrice.indptr[i + 1]
        colonnes = matrice.indices[debut:fin]
        valeurs = matrice.data[debut:fin]
        mask = colonnes > i
        if voisins_train[i]:
            mask &= ~np.isin(colonnes, list(voisins_train[i]))
        if not np.any(mask):
            continue
        c = colonnes[mask]
        s = valeurs[mask]
        if np.any(s <= 0):
            raise AssertionError('La matrice contient un score non positif')
        n_positifs += len(s)
        codes, counts = np.unique(s, return_counts=True)
        for code, count in zip(codes, counts):
            h[int(code)][0] += int(count)
        if voisins_test[i]:
            m = np.isin(c, list(voisins_test[i]))
            good_scores = s[m]
            vrais_positifs += int(m.sum())
            codes, counts = np.unique(good_scores, return_counts=True)
            for code, count in zip(codes, counts):
                h[int(code)][1] += int(count)
    return h, n_positifs, vrais_positifs


def courbe_depuis_groupes(groupes, candidats, vrais, max_k, positifs):
    if candidats <= 0 or not 0 < vrais <= candidats:
        raise ValueError('Ensembles de candidats/test invalides')
    groupe = {k: list(v) for k, v in groupes.items()}
    paires_positives, vraies_positives = positifs
    nb_zero = candidats - paires_positives
    hits_zero = vrais - vraies_positives
    if not (0 <= hits_zero <= nb_zero):
        raise AssertionError('Incohérence candidats / test / score nul')
    if nb_zero:
        groupe[0] = [nb_zero, hits_zero]
    k, hits, curve = 0, 0., []
    for score in sorted(groupe, reverse=True):
        count, good = groupe[score]
        upto = min(count, max_k - k)
        for t in range(1, upto + 1):
            kk = k + t
            exp = hits + t * good / count
            curve.append((kk, exp / kk, exp / vrais))
        k += count
        hits += good
        if k >= max_k:
            break
    return curve


def groupes_pour_dict(scores, test):
    h = defaultdict(lambda: [0, 0])
    hits = 0
    for pair, score in scores.items():
        if score <= 0 or not math.isfinite(score):
            raise AssertionError('L3 normalisé: score non positif/non fini')
        is_hit = pair in test
        h[score][0] += 1
        h[score][1] += int(is_hit)
        hits += int(is_hit)
    return h, (len(scores), hits)


def ecrire_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)


def figures(courbes, baseline, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib absent: graphiques non générés')
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, rows in courbes.items():
        ax.plot([r[2] * 100 for r in rows], [r[1] * 100 for r in rows], label=name)
    ax.axhline(baseline * 100, color='gray', linestyle='--', label='Hasard')
    ax.set(xlabel='Rappel (%)', ylabel='Précision (%)',
           title='HI-II-14 : longueurs de marche L2, L3, L4')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'courbe_precision_rappel_l2_l3_l4.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, rows in courbes.items():
        ax.plot([r[0] for r in rows], [r[1] * 100 for r in rows], label=name)
    ax.axhline(baseline * 100, color='gray', linestyle='--', label='Hasard')
    ax.set(xlabel='Nombre de prédictions k', ylabel='Précision@k (%)',
           title='HI-II-14 : précision selon k (L2, L3, L4)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'precision_selon_k_l2_l3_l4.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('results/l3_biologique'))
    p.add_argument('--output', type=Path, default=Path('results/l3_biologique/longueurs'))
    p.add_argument('--max-nodes', default='all', help='all ou entier (pour un essai rapide)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-k', type=int, default=20000)
    p.add_argument('--no-normalized', action='store_true', help='évite L3 normalisé optionnel')
    args = p.parse_args()
    train_full = lire_aretes(args.data / 'train_edges.tsv')
    test_full = lire_aretes(args.data / 'test_edges.tsv')
    nodes = lire_sommets(args.data / 'nodes.tsv')
    if (train_full & test_full):
        raise AssertionError('Fuite train/test')
    endpoints = {v for e in (train_full | test_full) for v in e}
    if not endpoints <= set(nodes):
        raise AssertionError('Sommets train/test absents de nodes.tsv')
    if args.max_nodes != 'all':
        n = int(args.max_nodes)
        if n <= 0 or n > len(nodes):
            raise ValueError('--max-nodes invalide')
        subset = set(np.random.default_rng(args.seed).choice(nodes, size=n, replace=False))
        nodes = sorted(subset)
        train = {e for e in train_full if e[0] in subset and e[1] in subset}
        test = {e for e in test_full if e[0] in subset and e[1] in subset}
    else:
        train, test = train_full, test_full
    if not train or not test:
        raise ValueError('Pas assez d\'arêtes train/test pour ce sous-graphe')
    a, ids, voisins_train = creer_matrice(nodes, train)
    n = len(nodes)
    candidats = n * (n - 1) // 2 - len(train)
    k = len(test)
    km = min(candidats, max(k, args.max_k))
    voisins_test = [set() for _ in nodes]
    for x, y in test:
        i, j = ids[x], ids[y]
        voisins_test[min(i, j)].add(max(i, j))
    baseline = len(test) / candidats
    title("HI-II-14 | L2 / L3 / L4", info=f"n={n:,} | k={k:,} | candidats={candidats:,}")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, all_curves = [], {}

    m = a
    for length in (2, 3, 4):
        t0 = time.perf_counter()
        m = (m @ a).tocsr()
        m.sum_duplicates()
        m.eliminate_zeros()
        m.sort_indices()
        dt = time.perf_counter() - t0
        h, positives, hits_pos = histogramme_scores(m, voisins_train, voisins_test)
        curve = courbe_depuis_groupes(h, candidats, k, km, (positives, hits_pos))
        row = curve[k - 1]
        name = f'L{length}_brut_A^{length}'
        all_curves[name] = curve
        summaries.append((name, len(nodes), len(train), k, positives, hits_pos,
                          f'{dt:.6f}', f'{row[1] * k:.4f}', f'{row[1] * 100:.6f}',
                          f'{row[2] * 100:.6f}', f'{row[1] / baseline:.4f}'))
        result(name, f"P@{k} = {100*row[1]:.2f}%", seconds=dt)

    # Important: L3 normalisé repose sur le même train/test mais une autre
    # pondération. Ses résultats ne doivent pas être confondus avec A^3 brut.
    if not args.no_normalized:
        del m, a
        gc.collect()
        graph = from_edges(train, nodes=nodes)
        t0 = time.perf_counter()
        scores = l3_optimise(graph)
        dt = time.perf_counter() - t0
        h, pos = groupes_pour_dict(scores, test)
        curve = courbe_depuis_groupes(h, candidats, k, km, pos)
        row = curve[k-1]
        name = 'L3_normalise'
        all_curves[name] = curve
        summaries.append((name, len(nodes), len(train), k, pos[0], pos[1],
                          f'{dt:.6f}', f'{row[1]*k:.4f}', f'{row[1]*100:.6f}',
                          f'{row[2]*100:.6f}', f'{row[1]/baseline:.4f}'))
        result(name, f"P@{k} = {100*row[1]:.2f}%", seconds=dt)
        del scores, graph

    ecrire_csv(args.output / 'resume_longueurs.csv',
               ['methode','sommets','train','test','scores_positifs','test_scores_positifs',
                'temps_calcul_s','vrais_top_k_esperance','precision_k_pct','rappel_k_pct','facteur_hasard'],
               summaries)
    ecrire_csv(args.output / 'courbes_longueurs.csv',
               ['methode','k','precision_attendue','rappel_attendu'],
               ((name, kk, f'{p:.12g}', f'{r:.12g}')
                for name, rows in all_curves.items() for kk, p, r in rows))
    figures(all_curves, baseline, args.output)
    proto = dict(source='HI-II-14', n=len(nodes), train=len(train), test=len(test),
                 candidats=candidats, baseline_precision=baseline,
                 definition='L2/L3/L4 bruts = (A^ell)[x,y], marches avec retours en arriere',
                 autre='L3_normalise = somme des chemins de longueur 3 pondérés 1/sqrt(dU*dV)',
                 split='Même train/test pour toutes les mesures; sous-échantillonnage induit si max-nodes!=all',
                 ties='Départage aléatoire uniforme en espérance y compris score nul',
                 limites='Une seule graine; présence de protéines isolées; aucune preuve biologique directe')
    (args.output / 'protocole_longueurs.json').write_text(json.dumps(proto, indent=2, ensure_ascii=False), encoding='utf-8')
    done(args.output)


if __name__ == '__main__':
    main()
