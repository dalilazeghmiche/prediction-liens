from itertools import combinations
from math import sqrt
from collections import defaultdict


def l3_naif(graphe):
    """
    graphe : dict[sommet, set[voisins]]
    Graphe simple, non orienté et non pondéré.

    Retour : dict[(x, y), score_L3]
    """
    scores = {}
    sommets = sorted(graphe)
    degres = {u: len(graphe[u]) for u in sommets}

    for x, y in combinations(sommets, 2):

        # On prédit uniquement les liens absents.
        if y in graphe[x]:
            continue

        score = 0.0

        # Chemins de longueur 3 : x -> u -> v -> y
        for u in graphe[x]:
            for v in graphe[u]:
                if y in graphe[v]:
                    score += 1 / sqrt(degres[u] * degres[v])

        if score > 0:
            scores[(x, y)] = score

    return scores


def l3_optimise(graphe, coeur=None):
    """
    Calcule L3 sur un graphe simple non orienté.

    graphe : dict[sommet, set[voisins]]
    coeur  : sommets autorisés comme extrémités des prédictions.
             Si None, tous les sommets sont candidats.

    Les sommets intermédiaires peuvent être hors du coeur.

    Retour : dict[(x, y), score_L3]
    """

    sommets = set(graphe)

    if coeur is None:
        coeur = sommets
    else:
        coeur = set(coeur)

    if not coeur <= sommets:
        raise ValueError("Le coeur contient des sommets inconnus")

    degres = {u: len(graphe[u]) for u in graphe}

    # Voisins utilisables comme extrémités d'une prédiction
    voisins_coeur = {
        u: graphe[u] & coeur
        for u in graphe
    }

    scores = defaultdict(float)

    # Chaque arête centrale U-V est parcourue une seule fois
    for u in graphe:
        for v in graphe[u]:

            if u >= v:
                continue

            contribution = 1.0 / sqrt(degres[u] * degres[v])

            # Chemins X-U-V-Y
            for x in voisins_coeur[u]:
                for y in voisins_coeur[v]:

                    # Pas de boucle
                    if x == y:
                        continue

                    # Ne pas prédire une arête déjà existante
                    if y in graphe[x]:
                        continue

                    # Paire non orientée
                    paire = (x, y) if x < y else (y, x)

                    scores[paire] += contribution

    return dict(scores)