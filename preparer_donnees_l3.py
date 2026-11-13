"""Prépare un découpage reproductible du réseau biologique HI-II-14.

À lancer depuis la racine du dépôt prediction-liens :
    python preparer_donnees_l3.py --data data/raw/HI-II-14.tsv --fraction 0.5 --seed 42

Ne calcule aucun score L3 : ce script prépare seulement les données.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from src.graph_io import from_edges, num_edges
from src.split import random_edge_split

_ENSG = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.IGNORECASE)


def lire_hi_ii_14(chemin):
    """Lit les deux premières colonnes TSV d'Ensembl IDs, nettoie boucles/doublons.

    Retourne (adjacence, statistiques). Les identifiants sont des chaînes.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier biologique introuvable : {chemin}")

    paires = set()
    sommets = set()
    lignes_donnees = 0
    doublons = 0
    boucles = 0
    entete_ignoree = False

    with chemin.open("r", encoding="utf-8-sig") as f:
        for numero, ligne in enumerate(f, 1):
            contenu = ligne.strip()
            if not contenu or contenu.startswith("#"):
                continue
            colonnes = contenu.split("\t")
            if len(colonnes) < 2:
                raise ValueError(f"Ligne {numero} : au moins deux colonnes TSV attendues")
            u, v = colonnes[0].strip(), colonnes[1].strip()

            # Certains exports TSV possèdent une ligne de noms de colonnes.
            if not (_ENSG.fullmatch(u) and _ENSG.fullmatch(v)):
                if lignes_donnees == 0 and not entete_ignoree and not paires:
                    if any(mot in (u + ' ' + v).lower() for mot in
                           ('gene', 'protein', 'interactor', 'ensembl', 'source', 'target')):
                        entete_ignoree = True
                        continue
                raise ValueError(
                    f"Ligne {numero} : identifiants Ensembl ENSG attendus, obtenus {u!r} et {v!r}. "
                    "Vérifie que le fichier téléchargé est bien HI-II-14.tsv et non une page HTML."
                )

            lignes_donnees += 1
            sommets.update((u, v))
            if u == v:
                boucles += 1
                continue
            paire = (u, v) if u < v else (v, u)
            if paire in paires:
                doublons += 1
            paires.add(paire)

    if not paires:
        raise ValueError("Aucune interaction exploitable dans le fichier")

    adj = from_edges(paires, nodes=sommets)
    infos = {
        'lignes_donnees': lignes_donnees,
        'boucles_ignorees': boucles,
        'doublons_ignores': doublons,
        'sommets': len(adj),
        'aretes_uniques': num_edges(adj),
        'entete_ignoree': entete_ignoree,
    }
    return adj, infos


def ecrire_aretes(path, aretes):
    path = Path(path)
    with path.open('w', encoding='utf-8', newline='') as f:
        for u, v in sorted(aretes):
            f.write(f'{u}\t{v}\n')


def principales_aretes(adj):
    return {(u, v) if u < v else (v, u)
            for u in adj for v in adj[u] if u != v}


def preparer(chemin, sortie, fraction, seed):
    if not (0.0 < fraction < 1.0):
        raise ValueError('--fraction doit être strictement entre 0 et 1')

    adj, infos = lire_hi_ii_14(chemin)
    split = random_edge_split(adj, test_fraction=fraction, seed=seed)
    train = principales_aretes(split.train_adj)
    test = split.new_edges
    originales = principales_aretes(adj)

    # Contrôles pour empêcher une fuite d'information entre train et test.
    assert not (train & test)
    assert train | test == originales
    assert set(split.train_adj) == set(adj)
    assert split.num_candidates == len(adj) * (len(adj) - 1) // 2 - len(train)

    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    ecrire_aretes(sortie / 'train_edges.tsv', train)
    ecrire_aretes(sortie / 'test_edges.tsv', test)
    # Conserver aussi les sommets sans arête après le nettoyage :
    # les reconstruire uniquement depuis train/test les ferait disparaître.
    (sortie / 'nodes.tsv').write_text(
        ''.join(f'{u}\n' for u in sorted(adj)), encoding='utf-8'
    )

    sha = hashlib.sha256(Path(chemin).read_bytes()).hexdigest()
    rapport = {
        'source': 'HI-II-14 — https://interactome-atlas.org/download',
        'fichier_source': str(chemin),
        'sha256_source': sha,
        'graphe': 'simple, non orienté, non pondéré',
        'seed': seed,
        'fraction_test_demandee': fraction,
        'nettoyage': infos,
        'sommets_total': len(adj),
        'sommets_isoles_apres_nettoyage': sum(not adj[u] for u in adj),
        'aretes_train': len(train),
        'aretes_test': len(test),
        'paires_candidates': split.num_candidates,
        'precision_hasard': len(test) / split.num_candidates,
        'sommets_isoles_dans_train': sum(not voisins for voisins in split.train_adj.values()),
        'note': 'Masquage aléatoire des arêtes, pas de datation ni de preuve expérimentale de liens absents.',
    }
    (sortie / 'protocole.json').write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print('=== Préparation du réseau biologique HI-II-14 ===')
    print(f"Articles/ligne d'interactions : {infos['lignes_donnees']:,}")
    print(f"Sommets : {len(adj):,} | arêtes uniques : {len(originales):,}")
    print(f"Boucles ignorées : {infos['boucles_ignorees']:,} | doublons ignorés : {infos['doublons_ignores']:,}")
    print(f'Apprentissage : {len(train):,} arêtes | Test : {len(test):,} arêtes')
    print(f'Sommets isolés en apprentissage : {rapport["sommets_isoles_dans_train"]:,}')
    print(f'Paires candidates : {split.num_candidates:,}')
    print(f'Précision du hasard (attendue) : {100 * rapport["precision_hasard"]:.6f} %')
    print('Vérifications : aucun chevauchement train/test, ensemble original reconstitué : OK')
    print(f'Fichiers : {sortie}')
    return rapport


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Préparer le réseau HI-II-14 pour comparer L3/CN/AA.')
    p.add_argument('--data', default='data/raw/HI-II-14.tsv', help='Chemin du TSV HuRI')
    p.add_argument('--out', default='results/l3_biologique', help='Dossier des résultats')
    p.add_argument('--fraction', type=float, default=0.5, help='Fraction des arêtes masquées (défaut 0.5)')
    p.add_argument('--seed', type=int, default=42, help='Graine du tirage (défaut 42)')
    a = p.parse_args()
    preparer(a.data, a.out, a.fraction, a.seed)
