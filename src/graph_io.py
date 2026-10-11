""" Lecture de graphes non orientés et construction des listes d'adjacence.

Représentation choisie: un dictionnaire {sommet : ensemble de voisins}.
- mémoire en O(n + m), n = nombre de sommets, m = nombre d'aretes
- test << (u,v) est-elle une arete ? >> en O(1) en moyenne (ensembles Python).
"""

from collections import defaultdict


class EdgeListFormatError(ValueError):
    """Ligne mal formée dans un fichier de liste d'arêtes."""


def from_edges(edges, nodes=None):
    """Construit les listes d'adjacence à partir d'un itérable de paires (u, v).

    Le graphe est non orienté et simple : les boucles (u, u) sont ignorées
    et les arêtes en double ne sont comptées qu'une fois.
    `nodes` (optionnel) : sommets à ajouter même s'ils sont isolés.
    """
    adj = defaultdict(set)
    if nodes is not None:
        for u in nodes:
            adj[u]  # crée une entrée vide pour un sommet isolé
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return dict(adj)


def _iter_rows(path, comment, sep, min_cols, strict):
    """Renvoie (numéro de ligne, colonnes) pour chaque ligne utile du fichier."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            parts = line.split(sep)
            if len(parts) < min_cols:
                if strict:
                    raise EdgeListFormatError(
                        f"{path}, ligne {lineno} : {min_cols} colonnes attendues, "
                        f"{len(parts)} trouvée(s) : {line!r}")
                continue
            yield lineno, parts


def _convert(value, node_type, path, lineno, what):
    try:
        return node_type(value)
    except ValueError as e:
        raise EdgeListFormatError(
            f"{path}, ligne {lineno} : {what} {value!r} non convertible "
            f"avec {node_type.__name__}") from e


def read_edge_list(path, comment="#", sep=None, node_type=str, strict=True):
    """Lit un fichier « liste d'arêtes » (une arête par ligne : « u v »).

    - les lignes vides et celles commençant par `comment` sont ignorées ;
    - `sep=None` découpe sur n'importe quel espace ou tabulation ;
    - les colonnes au-delà des deux premières sont ignorées ici
      (pour les dates, utiliser `read_temporal_edges`) ;
    - `node_type` convertit les identifiants (str par défaut, int possible) ;
    - `strict=True` : une ligne mal formée lève EdgeListFormatError avec son
      numéro de ligne ; `strict=False` : elle est ignorée silencieusement.
    """
    def edges():
        for lineno, parts in _iter_rows(path, comment, sep, 2, strict):
            yield (_convert(parts[0], node_type, path, lineno, "sommet"),
                   _convert(parts[1], node_type, path, lineno, "sommet"))

    return from_edges(edges())


def read_temporal_edges(path, time_col=2, comment="#", sep=None,
                        node_type=str, time_type=int, strict=True):
    """Lit un fichier d'arêtes datées « u v ... t » pour le protocole temporel.

    Retourne la liste des triplets (u, v, t), sans fusionner les doublons :
    une même paire peut apparaître à plusieurs dates (plusieurs articles).
    `time_col` : indice (à partir de 0) de la colonne contenant la date.
    """
    triples = []
    for lineno, parts in _iter_rows(path, comment, sep, time_col + 1, strict):
        u = _convert(parts[0], node_type, path, lineno, "sommet")
        v = _convert(parts[1], node_type, path, lineno, "sommet")
        t = _convert(parts[time_col], time_type, path, lineno, "date")
        triples.append((u, v, t))
    return triples


def num_nodes(adj):
    return len(adj)


def num_edges(adj):
    return sum(len(nbrs) for nbrs in adj.values()) // 2


def has_edge(adj, u, v):
    return v in adj.get(u, ())
