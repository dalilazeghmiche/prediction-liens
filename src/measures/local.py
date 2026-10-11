"""Mesures locales de prédiction de liens.

Pour l'instant : voisins communs (Common Neighbours, CN).

    CN(x, y) = |Γ(x) ∩ Γ(y)|

Idée clé pour l'efficacité : une paire (x, y) a CN > 0 si et seulement si
x et y ont au moins un voisin commun w, c'est-à-dire s'ils sont à distance
≤ 2. Au lieu de tester les n² paires, on parcourt chaque sommet w et chaque
paire de ses voisins : chaque passage ajoute 1 au CN de la paire.
Coût : Σ_w d(w)² au lieu de n² (même boucle que l'algorithme naïf de
comptage de triangles du cours Fouille de motifs).
"""

from itertools import combinations


def _pair(u, v):
    """Clé canonique d'une paire non orientée : (u, v) avec u < v."""
    return (u, v) if u < v else (v, u)


def common_neighbors(adj, x, y):
    """CN d'une seule paire : taille de l'intersection des voisinages.

    On parcourt le plus petit voisinage et on teste l'appartenance dans
    l'autre (ensemble) : coût O(min(d(x), d(y))).
    """
    nx, ny = adj.get(x, set()), adj.get(y, set())
    if len(nx) > len(ny):
        nx, ny = ny, nx
    return sum(1 for w in nx if w in ny)


def cn_scores(adj, nodes=None):
    """CN de toutes les paires NON reliées ayant au moins un voisin commun.

    - `nodes` : si fourni (ex. ensemble des sommets cœur), on ne garde que
      les paires dont les deux extrémités sont dans `nodes`.
    - Les paires absentes du résultat ont un score de 0.

    Retourne un dictionnaire {(x, y): CN} avec x < y.
    """
    keep = None if nodes is None else set(nodes)
    scores = {}
    for w, nbrs in adj.items():
        cand = nbrs if keep is None else [u for u in nbrs if u in keep]
        for x, y in combinations(cand, 2):
            if y in adj[x]:          # déjà reliés : pas un lien à prédire
                continue
            key = _pair(x, y)
            scores[key] = scores.get(key, 0) + 1
    return scores


def rank(scores):
    """Trie les paires par score décroissant (égalités départagées par la
    paire elle-même, pour que le classement soit reproductible)."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
