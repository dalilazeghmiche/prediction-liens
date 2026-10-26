"""Évaluation des prédictions : précision, rappel, comparaison au hasard.

On prédit les k premières paires du classement :
    précision@k = (bonnes prédictions parmi les k premières) / k
    rappel@k    = (bonnes prédictions parmi les k premières) / |liens à prédire|
En faisant varier k on obtient la courbe rappel-précision demandée.

Comme dans [2], on regarde aussi k = n (n = nombre de liens à prédire) et
le facteur d'amélioration par rapport au prédicteur aléatoire, dont la
précision attendue est |liens à prédire| / |paires candidates|.

Remarque : les mesures locales ne classent que les paires de score > 0.
Les autres sont ex æquo en dernier ; la courbe s'arrête donc à la fin du
classement (le rappel maximal atteint est une information en soi).
"""


def precision_recall_curve(ranked, positives, k_max=None):
    """ranked    : liste de paires (x, y) triées, ou de ((x, y), score)
                   comme renvoyé par rank() ;
    positives : ensemble des paires (x, y), x < y, réellement apparues.
    Retourne trois listes : ks, précisions, rappels (k = 1 .. k_max)."""
    if not positives:
        raise ValueError("aucun lien à prédire : évaluation impossible")
    total = len(positives)
    k_max = len(ranked) if k_max is None else min(k_max, len(ranked))
    ks, precisions, recalls = [], [], []
    hits = 0
    for k in range(1, k_max + 1):
        item = ranked[k - 1]
        pair = item[0] if isinstance(item[0], tuple) else item
        if pair in positives:
            hits += 1
        ks.append(k)
        precisions.append(hits / k)
        recalls.append(hits / total)
    return ks, precisions, recalls


def precision_at(ranked, positives, k):
    """Précision des k premières prédictions (k = n dans le protocole de [2])."""
    if k <= 0:
        raise ValueError("k doit être > 0")
    _, precisions, _ = precision_recall_curve(ranked, positives, k_max=k)
    if not precisions:
        return 0.0
    # Si le classement contient moins de k paires, les paires manquantes
    # (score 0) comptent comme de mauvaises prédictions.
    hits = precisions[-1] * len(precisions)
    return hits / k


def random_precision(num_positives, num_candidates):
    """Précision attendue d'un prédicteur qui tire les paires au hasard."""
    return num_positives / num_candidates


def improvement_factor(ranked, positives, num_candidates):
    """Facteur d'amélioration sur le hasard à k = n, comme dans [2]."""
    n = len(positives)
    return precision_at(ranked, positives, n) / random_precision(n, num_candidates)