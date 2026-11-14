# AAGA — Prédiction de liens

CN, Adamic-Adar et RWR sur **hep-th** ; CN, AA et L3 sur **HI-II-14**.
Les algorithmes sont dans `src/` ; les scripts Python d'expériences sont à la racine.

## Installation

Python 3.11+ ; Bash (Git Bash sous Windows). Depuis la racine du projet :

```bash
python -m pip install -r requirements.txt
```

Placer dans `data/raw/` : `hep-th-abs.tar.gz`, `hep-th-slacdates.tar.gz` et `HI-II-14.tsv`.

## Exécution

```bash
bash run.sh --quick                 # tests + préparation des données
bash run.sh --social                # partie 1 complète
bash run.sh --bio                   # partie 2 complète
bash run.sh --all --repeats 3       # toutes les expériences
bash run.sh --social --sizes 500 --repeats 1  # essai court
bash run.sh --help                  # options
```

Options Bash : un seul mode parmi `--all` (défaut), `--social`, `--bio`, `--tests`, `--quick` ; `--repeats N` (défaut 3) et `--sizes S` (défaut `500,1000,2000,all`, `all` = graphe complet).

## Expériences et résultats

| Script Python | Résultat dans `results/` | Options principales |
|---|---|---|
| `verifier_protocole_partie1.py`, `audit_hepth_article2.py` | Bilan console hep-th | aucune |
| `evaluation_cn_aa_hepth.py` | `evaluation_cn_aa_hepth/` : CSV Top-k et résumé | aucune |
| `evaluation_rwr_hepth.py` | `evaluation_rwr_hepth/` : CSV Top-k et résumé | `--alpha`, `--tol`, `--max-iter`, `--block-size` |
| `courbes_precision_rappel_hepth.py` | `courbes_precision_rappel_hepth/` : CSV, PNG, protocole | mêmes options RWR |
| `comparaison_rwr_original_symetrise_hepth.py` | `comparaison_rwr_original_symetrise_hepth/` : CSV, PNG | mêmes options RWR ; `--max-k` |
| `benchmarks_cn_aa_rwr_hepth.py` | `benchmarks_cn_aa_rwr_hepth/` : temps, mémoire, PNG | `--sizes`, `--repeats`, `--seed`, options RWR |
| `verifier_resultats_social.py` | Contrôle console des sorties sociales | aucune |
| `preparer_donnees_l3.py` | `l3_biologique/` : train/test, sommets, protocole | `--data`, `--out`, `--fraction`, `--seed` |
| `evaluer_l3_bio.py` | `l3_biologique/evaluation/` : CSV, PNG | `--data`, `--out`, `--max-k` |
| `benchmarks_l3_bio.py` | `l3_biologique/benchmarks/` : CSV, PNG | `--sizes`, `--repeats`, `--naive-max` |
| `comparer_l2_l3_l4_bio.py` | `l3_biologique/longueurs/` : CSV, PNG | `--max-nodes`, `--max-k`, `--no-normalized` |
| `robustesse_l3_bio.py` | `l3_biologique/robustesse/` : CSV, PNG | `--seeds`, `--fraction`, `--no-plots` |

Lancer chaque script individuellement avec `python nom_du_script.py --help` pour voir ses valeurs par défaut et ses autres options.

**Protocole :** hep-th train 1994–1996, test 1997–1999, cœur 3+3 ; RWR α=0,15 ; HI-II-14 masquage 50 % graine 42, robustesse graines 42–46. Ex æquo et scores nuls traités en espérance dans les courbes. Les sous-graphes de benchmark servent à mesurer les performances, pas à estimer la précision sur le corpus complet. Le Bash fixe `PYTHONHASHSEED=0` pour limiter les petites variations des classements dues à l’ordre des ensembles Python. Les mesures de mémoire RSS sont approximatives.
