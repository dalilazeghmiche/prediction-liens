# AAGA — Prédiction de liens

Projet AAGA — Sujet 8, M2 STL, Sorbonne Université.

Implémentation et évaluation de quatre mesures de prédiction de liens :

- **Réseau social hep-th** : Common Neighbors (CN), Adamic-Adar (AA) et Random Walk with Restart (RWR).
- **Réseau biologique HI-II-14** : CN, AA et L3.

Les algorithmes sont dans `src/`, les tests dans `tests/` et les scripts Python d'expérimentation à la racine du projet.

## Installation

**Prérequis :** Python 3.11 ou supérieur et Bash (Git Bash sous Windows).

Depuis la racine du projet :

```bash
python -m pip install -r requirements.txt
```

Placer les fichiers de données suivants dans `data/raw/` :

| Fichier                   | Description                                        |
| ------------------------- | -------------------------------------------------- |
| `hep-th-abs.tar.gz`       | Archive des résumés et auteurs des articles hep-th |
| `hep-th-slacdates.tar.gz` | Dates des articles hep-th                          |
| `HI-II-14.tsv`            | Réseau d'interactions entre protéines humaines     |

**Provenance :** les données hep-th proviennent du corpus arXiv utilisé pour les expériences de prédiction de liens ; HI-II-14 est un réseau d'interactions protéiques humaines étudié dans les travaux de Kovács et al. (2019). Les références scientifiques et les détails du protocole figurent dans le rapport.

## Exécution

Le fichier `run.sh` permet de lancer les expériences des deux parties.

```bash
bash run.sh --tests                         # 67 tests unitaires
bash run.sh --quick                         # tests et vérifications rapides
bash run.sh --social                        # expériences sociales complètes
bash run.sh --bio                           # expériences biologiques complètes
bash run.sh --all --repeats 3               # toutes les expériences
bash run.sh --social --sizes 500 --repeats 1 # benchmarks limités à 500 sommets
bash run.sh --help                          # aide et options
```

**Options du script Bash :**

| Option        | Description                                                                   |
| ------------- | ----------------------------------------------------------------------------- |
| `--all`       | Lance toutes les expériences (mode par défaut)                                |
| `--social`    | Lance les expériences sur hep-th                                              |
| `--bio`       | Lance les expériences sur HI-II-14                                            |
| `--tests`     | Exécute les 67 tests unitaires                                                |
| `--quick`     | Lance les tests et contrôles rapides                                          |
| `--repeats N` | Nombre de répétitions des benchmarks (3 par défaut)                           |
| `--sizes S`   | Tailles des sous-graphes pour les benchmarks (par défaut `500,1000,2000,all`) |
| `--help`      | Affiche les options disponibles                                               |

Les modes `--all`, `--social`, `--bio`, `--tests` et `--quick` sont exclusifs.

L'option `--sizes` concerne les benchmarks : elle ne réduit pas la taille des jeux de données utilisés pour les évaluations principales.

## Interface graphique (facultative)

Le fichier `interface.py` fournit une interface interactive développée avec **Streamlit**.

Elle permet de choisir le jeu de données, les paramètres expérimentaux et les algorithmes, puis d'explorer les graphes et les prédictions.

**Fonctionnalités principales :**

- **Jeu de données :** vue d'ensemble du protocole et du découpage apprentissage/test.
- **Structure du réseau :** distribution des degrés, triangles, clustering, transitivité, composantes et assortativité.
- **Liens à prédire :** analyse des distances entre les sommets dans le graphe d'apprentissage.
- **Exploration d'un sommet :** visualisation du voisinage à un ou deux sauts, des liens masqués et des prédictions correctes ou incorrectes.
- **Résultats :** précision, rappel et comparaison des algorithmes.
- **Courbes :** graphiques précision–rappel.
- **Prédictions :** classement des meilleures paires avec leurs scores.

Installation et lancement :

```bash
python -m pip install -r requirements-interface.txt
python -m streamlit run interface.py
```

L'interface est accessible à l'adresse `http://localhost:8501`.

RWR est proposé uniquement sur hep-th. Les paramètres par défaut correspondent à ceux du rapport ; l'interface permet également d'explorer d'autres configurations.

**Les résultats de référence du rapport sont ceux des expériences exécutées avec `run.sh`.** L'interface constitue un outil complémentaire d'exploration et de visualisation.

## Expériences et résultats

| Script Python                                              | Sorties                                                              | Options principales                              |
| ---------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------ |
| `verifier_protocole_partie1.py`, `audit_hepth_article2.py` | Bilan console du protocole hep-th                                    | Aucune                                           |
| `evaluation_cn_aa_hepth.py`                                | `results/evaluation_cn_aa_hepth/` : CSV Top-k et résumé              | Aucune                                           |
| `evaluation_rwr_hepth.py`                                  | `results/evaluation_rwr_hepth/` : CSV Top-k et résumé                | `--alpha`, `--tol`, `--max-iter`, `--block-size` |
| `courbes_precision_rappel_hepth.py`                        | `results/courbes_precision_rappel_hepth/` : CSV, PNG et protocole    | Options RWR                                      |
| `comparaison_rwr_original_symetrise_hepth.py`              | `results/comparaison_rwr_original_symetrise_hepth/` : CSV et PNG     | Options RWR, `--max-k`                           |
| `benchmarks_cn_aa_rwr_hepth.py`                            | `results/benchmarks_cn_aa_rwr_hepth/` : temps, mémoire et graphiques | `--sizes`, `--repeats`, `--seed`, options RWR    |
| `verifier_resultats_social.py`                             | Contrôle console des résultats sociaux                               | Aucune                                           |
| `preparer_donnees_l3.py`                                   | `results/l3_biologique/` : train/test, sommets et protocole          | `--data`, `--out`, `--fraction`, `--seed`        |
| `evaluer_l3_bio.py`                                        | `results/l3_biologique/evaluation/` : CSV et PNG                     | `--data`, `--out`, `--max-k`                     |
| `benchmarks_l3_bio.py`                                     | `results/l3_biologique/benchmarks/` : CSV et PNG                     | `--sizes`, `--repeats`, `--naive-max`            |
| `comparer_l2_l3_l4_bio.py`                                 | `results/l3_biologique/longueurs/` : CSV et PNG                      | `--max-nodes`, `--max-k`, `--no-normalized`      |
| `robustesse_l3_bio.py`                                     | `results/l3_biologique/robustesse/` : CSV et PNG                     | `--seeds`, `--fraction`, `--no-plots`            |

Pour consulter les options détaillées et leurs valeurs par défaut :

```bash
python nom_du_script.py --help
```

Les fichiers CSV contiennent les résultats numériques des expériences ; les PNG sont utilisés pour les figures et les comparaisons du rapport.

## Protocole expérimental

**Partie 1 — hep-th**

- Apprentissage : 1994–1996.
- Test : 1997–1999.
- Cœur : auteurs ayant publié au moins trois articles dans chacune des deux périodes.
- RWR : probabilité de redémarrage `alpha = 0.15`.

**Partie 2 — HI-II-14**

- Masquage aléatoire de 50 % des interactions.
- Graine principale : `42`.
- Robustesse : graines `42,43,44,45,46`.
- Les sommets isolés sont conservés lors du découpage.

**Évaluation et reproductibilité**

- Les courbes précision–rappel utilisent un départage uniforme en espérance pour les ex æquo et les scores nuls.
- Les benchmarks utilisent des sous-graphes pour mesurer les temps et la mémoire, sans modifier les résultats d'évaluation sur les corpus complets.
- Le script Bash fixe `PYTHONHASHSEED=0` pour stabiliser l'ordre de parcours des ensembles Python entre les exécutions.
- Les mesures de mémoire RSS sont approximatives et peuvent manquer des pics de courte durée.

Les résultats expérimentaux sont enregistrés dans `results/`.
