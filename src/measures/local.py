"""Mesures locales de prédiction de liens.

- Voisins communs (Common Neighbours, CN) :
      CN(x, y) = |Γ(x) ∩ Γ(y)|
- Adamic-Adar (AA) :
      AA(x, y) = Σ_{w ∈ Γ(x) ∩ Γ(y)} 1 / log d(w)
  Un voisin commun peu connecté pèse plus qu'un sommet très connecté.
 
Idée clé pour l'efficacité : une paire (x, y) a un score > 0 si et seulement
si x et y ont au moins un voisin commun w, c'est-à-dire s'ils sont à
distance 2. Au lieu de tester les n² paires, on parcourt chaque sommet w et
chaque paire de ses voisins : chaque passage ajoute la contribution de w au
score de la paire (1 pour CN, 1/log d(w) pour AA).
Coût : Σ_w d(w)² au lieu de n² (même boucle que l'algorithme naïf de
comptage de triangles du cours Fouille de motifs).
"""

from itertools import combinations
import math


def _pair(u, v):
    """Clé canonique d'une paire non orientée : (u, v) avec u < v."""
    return (u, v) if u < v else (v, u)



def _common(adj, x, y):
    """Retourne les voisins communs de x et y."""
    nx, ny = adj.get(x, set()), adj.get(y, set())

    if len(nx) > len(ny):
        nx, ny = ny, nx

    return [w for w in nx if w in ny]


def common_neighbors(adj, x, y):
    """Calcule le nombre de voisins communs."""
    return len(_common(adj, x, y))


def _pair_scores(adj, weight, nodes=None):
    """Boucle commune à CN et AA.
 
    Pour chaque sommet w et chaque paire (x, y) de ses voisins NON reliés,
    ajoute weight(w) au score de (x, y).
    - `nodes` : si fourni (ex. sommets cœur), on ne garde que les paires
      dont les deux extrémités sont dans `nodes` ; le voisin commun w, lui,
      peut être n'importe quel sommet.
    Retourne {(x, y): score} avec x < y ; les paires absentes valent 0.
    """
    keep = None if nodes is None else set(nodes)
    scores = {}

    for w, nbrs in adj.items():
        cand = nbrs if keep is None else [
            u for u in nbrs if u in keep
        ]

        if len(cand) < 2:
            continue

        c = weight(w)

        for x, y in combinations(cand, 2):
            if y in adj[x]:
                continue

            key = _pair(x, y)
            scores[key] = scores.get(key, 0) + c

    return scores


def cn_scores(adj, nodes=None):
    """Calcule les scores Common Neighbours."""
    return _pair_scores(adj, lambda w: 1, nodes)


def _aa_weight(adj, w):
    return 1.0 / math.log(len(adj[w]))


def adamic_adar(adj, x, y):
    """Calcule le score AA d'une paire."""
    return sum(
        _aa_weight(adj, w)
        for w in _common(adj, x, y)
    )


def aa_scores(adj, nodes=None):
    """Calcule les scores Adamic-Adar."""
    return _pair_scores(
        adj,
        lambda w: _aa_weight(adj, w),
        nodes
    )



def rank(scores):
    """Trie les paires par score décroissant (égalités départagées par la
    paire elle-même, pour que le classement soit reproductible)."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
