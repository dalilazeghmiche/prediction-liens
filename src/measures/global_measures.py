"""Mesures globales de prédiction de liens.

RWR — marche aléatoire avec redémarrage (« rooted PageRank » dans [2]).

Un marcheur part de x. À chaque pas :
  - avec probabilité α, il revient en x (redémarrage) ;
  - avec probabilité 1 - α, il va chez un voisin choisi au hasard.
r_x(y) = probabilité, à l'état stationnaire, de trouver le marcheur en y.

C'est le PageRank du cours (puissance itérée + « évaporation »), sauf que
la téléportation renvoie toujours vers x au lieu d'un sommet au hasard.
Mise à jour (puissance itérée) :
    r ← α·e_x + (1 - α)·P·r       avec P = A·D⁻¹ (matrice de transition)

r_x(y) n'est pas symétrique. Pour classer des paires NON orientées on
utilise score(x, y) = r_x(y) + r_y(x) (choix courant, cf. Lü et Zhou 2011),
à justifier dans le rapport.

Efficacité : on ne stocke jamais de matrice n × n. On fait des produits
« matrice creuse × bloc de vecteurs » (cours PageRank, multiplication
matrice creuse-vecteur), pour un bloc de sources à la fois.
Coût : O(itérations × m × |sources|), mémoire O(m + n × taille_bloc).

Deux implémentations :
  - rwr_from        : version simple en Python pur, pour une source
                      (sert de référence dans les tests) ;
  - rwr_scores      : version efficace NumPy/SciPy pour toutes les paires.
"""

import numpy as np
import scipy.sparse as sp

from src.split import pair_key


# ------------------------------------------------------------- référence

def rwr_from(adj, source, alpha=0.15, tol=1e-12, max_iter=10_000):
    """Vecteur RWR {sommet: probabilité} depuis `source` (Python pur).

    Un sommet sans voisin (cul-de-sac) renvoie le marcheur en `source`
    (même idée que le traitement des culs-de-sac du PageRank).
    """
    r = {source: 1.0}
    for _ in range(max_iter):
        new = {source: alpha}
        lost = 0.0
        for u, p in r.items():
            nbrs = adj.get(u, ())
            if not nbrs:
                lost += p
                continue
            share = (1 - alpha) * p / len(nbrs)
            for v in nbrs:
                new[v] = new.get(v, 0.0) + share
        new[source] += (1 - alpha) * lost
        diff = sum(abs(new.get(k, 0.0) - r.get(k, 0.0))
                   for k in set(new) | set(r))
        r = new
        if diff < tol:
            break
    return r


# ------------------------------------------------------------- efficace

def _transition_matrix(adj):
    """Matrice de transition creuse P = A·D⁻¹ (colonnes stochastiques)."""
    nodes = list(adj)
    index = {u: i for i, u in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for u in nodes:
        d = len(adj[u])
        for v in adj[u]:
            rows.append(index[v])     # de u (colonne) vers v (ligne)
            cols.append(index[u])
            vals.append(1.0 / d)
    n = len(nodes)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return P, nodes, index


def rwr_matrix(adj, sources, alpha=0.15, tol=1e-10, max_iter=1000,
               block_size=256):
    """Pour chaque source x, le vecteur r_x sur tous les sommets.

    Retourne (R, nodes, index) : R[:, j] = r_{sources[j]}, matrice dense
    n × |sources|. Calcul par blocs de `block_size` sources.
    """
    P, nodes, index = _transition_matrix(adj)
    n = len(nodes)
    sources = list(sources)
    R = np.zeros((n, len(sources)))
    for start in range(0, len(sources), block_size):
        block = sources[start:start + block_size]
        b = len(block)
        src_idx = np.array([index[s] for s in block])
        E = np.zeros((n, b))
        E[src_idx, np.arange(b)] = 1.0
        X = E.copy()
        for _ in range(max_iter):
            X_new = (1 - alpha) * (P @ X)
            lost = 1.0 - X_new.sum(axis=0)          # α + masse des culs-de-sac
            X_new[src_idx, np.arange(b)] += lost    # tout revient à la source
            diff = np.abs(X_new - X).sum(axis=0).max()
            X = X_new
            if diff < tol:
                break
        R[:, start:start + b] = X
    return R, nodes, index


def rwr_scores(adj, nodes=None, alpha=0.15, tol=1e-10, max_iter=1000,
               block_size=256):
    """Score RWR symétrisé de toutes les paires NON reliées.

    - `nodes` : sommets cœur (par défaut : tous). Les marches se font sur
      tout le graphe, mais on ne note que les paires de sommets cœur.
    - score(x, y) = r_x(y) + r_y(x) ; seules les paires de score > 0
      (même composante connexe) sont renvoyées.
    Retourne {(x, y): score} avec x < y.
    """
    core = list(adj) if nodes is None else [u for u in nodes if u in adj]
    R, _, index = rwr_matrix(adj, core, alpha, tol, max_iter, block_size)
    rows = np.array([index[u] for u in core])
    S = R[rows, :]                # S[i, j] = r_{core[j]}(core[i])
    S = S + S.T                   # symétrisation
    scores = {}
    iu, ju = np.triu_indices(len(core), k=1)
    vals = S[iu, ju]
    for i, j, s in zip(iu[vals > 0], ju[vals > 0], vals[vals > 0]):
        x, y = core[i], core[j]
        if y in adj[x]:
            continue
        scores[pair_key(x, y)] = float(s)
    return scores