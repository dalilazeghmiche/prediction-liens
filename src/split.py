"""Construction des données d'apprentissage et de test.

1. Protocole temporel de Liben-Nowell et Kleinberg [2] (méthode principale) :
   - graphe d'apprentissage : co-auteurs des articles de la période d'apprentissage ;
   - sommets cœur : auteurs ayant au moins `k_train` articles pendant
     l'apprentissage ET au moins `k_test` articles pendant le test
     (dans [2] : k_train = k_test = 3) ;
   - liens à prédire : paires de sommets cœur qui collaborent pendant le
     test mais n'étaient PAS reliées pendant l'apprentissage.

2. Masquage aléatoire d'arêtes (solution de repli si les données ne sont
   pas datées) : on cache une fraction des arêtes, à retrouver.

Format d'entrée : une liste d'articles (date, [auteurs]).
Un fichier d'arêtes datées « u v t » se convertit avec
`papers_from_temporal_edges` (chaque arête = un article à deux auteurs).
"""

import random
from dataclasses import dataclass
from itertools import combinations

from src.graph_io import from_edges


def pair_key(u, v):
    """Clé canonique d'une paire non orientée : (u, v) avec u < v."""
    return (u, v) if u < v else (v, u)


def papers_from_temporal_edges(triples):
    """Convertit des arêtes datées (u, v, t) en articles (t, [u, v])."""
    return [(t, [u, v]) for u, v, t in triples]


def _in(t, period):
    return period[0] <= t <= period[1]


@dataclass
class Split:
    train_adj: dict        # graphe d'apprentissage {sommet: voisins}
    core: set              # sommets cœur
    new_edges: set         # paires (x, y), x < y, à prédire
    num_candidates: int    # nombre de paires cœur non reliées à l'apprentissage


def count_candidate_pairs(adj, core):
    """Nombre de paires de sommets cœur NON reliées dans `adj`.
    C'est le nombre total de paires parmi lesquelles on prédit ; il sert
    à calculer la précision du prédicteur aléatoire."""
    core = set(core)
    n = len(core)
    edges_in_core = sum(1 for u in core for v in adj.get(u, ())
                        if v in core) // 2
    return n * (n - 1) // 2 - edges_in_core


def temporal_split(papers, train, test, k_train=3, k_test=3):
    """Protocole de [2].

    papers : liste de (date, [auteurs])
    train, test : périodes (début, fin), bornes incluses, ex. (1994, 1996)
    """
    train_edges, train_authors = [], set()
    count_train, count_test = {}, {}
    test_pairs = set()

    for t, authors in papers:
        authors = list(dict.fromkeys(authors))   # retire les doublons
        if _in(t, train):
            train_authors.update(authors)
            for a in authors:
                count_train[a] = count_train.get(a, 0) + 1
            train_edges.extend(combinations(authors, 2))
        elif _in(t, test):
            for a in authors:
                count_test[a] = count_test.get(a, 0) + 1
            test_pairs.update(pair_key(u, v)
                              for u, v in combinations(authors, 2) if u != v)

    train_adj = from_edges(train_edges, nodes=train_authors)
    core = {a for a in train_authors
            if count_train.get(a, 0) >= k_train
            and count_test.get(a, 0) >= k_test}
    new_edges = {(u, v) for u, v in test_pairs
                 if u in core and v in core and v not in train_adj[u]}
    return Split(train_adj, core, new_edges,
                 count_candidate_pairs(train_adj, core))


def random_edge_split(adj, test_fraction=0.1, seed=0):
    """Repli sans dates : cache aléatoirement `test_fraction` des arêtes.
    Tous les sommets sont « cœur ». Reproductible grâce à `seed`."""
    edges = sorted({pair_key(u, v) for u in adj for v in adj[u]})
    rng = random.Random(seed)
    hidden = set(rng.sample(edges, int(round(test_fraction * len(edges)))))
    train_adj = from_edges((e for e in edges if e not in hidden),
                           nodes=adj.keys())
    core = set(adj)
    return Split(train_adj, core, hidden,
                 count_candidate_pairs(train_adj, core))