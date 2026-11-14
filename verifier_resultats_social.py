"""Vérifie la cohérence des CSV et graphiques de la partie sociale."""
from __future__ import annotations
import csv
import math
from pathlib import Path
from utils_experiences import title

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def rows_csv(folder: str, file: str):
    p = RESULTS / folder / file
    if not p.is_file():
        raise FileNotFoundError(f"Fichier manquant : {p.relative_to(ROOT)}")
    with p.open(newline='', encoding='utf-8-sig') as f:
        out = list(csv.DictReader(f))
    if not out:
        raise AssertionError(f"CSV vide : {p}")
    return out


def required_png(folder: str, file: str):
    p = RESULTS / folder / file
    if not p.is_file() or p.stat().st_size == 0:
        raise AssertionError(f"Figure manquante ou vide : {p.relative_to(ROOT)}")


def main():
    prefix='courbes_precision_rappel_hepth'
    rows=rows_csv(prefix,'synthese_precision_rappel.csv')
    algos={r['algorithme']:r for r in rows}
    required={'CN','AA','RWR_symetrise'}
    if set(algos) != required:
        raise AssertionError(f"Algorithmes attendus {required}, obtenus {set(algos)}")
    n_values={int(r['candidats']) for r in rows}
    k_values={int(r['nouveaux_liens']) for r in rows}
    if len(n_values)!=1 or len(k_values)!=1:
        raise AssertionError('Les trois algorithmes ne partagent pas le meme protocole')
    n=n_values.pop(); k=k_values.pop()
    if not (0<k<n): raise AssertionError('Nombre de candidats / de nouveaux liens incoherent')
    for r in rows:
        if int(r['k_reference']) != k:
            raise AssertionError(f"k non comparable : {r['algorithme']}")
        hits=int(r['tp_topk_deterministe'])
        if not 0<=hits<=k: raise AssertionError(f"TP invalides : {r['algorithme']}")
        p=float(r['precision_topk_deterministe'])
        if not math.isclose(p, hits/k, abs_tol=1e-10):
            raise AssertionError(f"Precision et TP non concordants : {r['algorithme']}")
        exp=float(r['tp_topk_exaequo_moyen'])
        if not 0<=exp<=k: raise AssertionError(f"TP esperes invalides : {r['algorithme']}")
        if int(r['liens_reels_score_positif'])>k:
            raise AssertionError('Nombre de positifs recuperables invalide')
    for png in ('courbe_precision_rappel_complete.png', 'courbe_precision_rappel_zoom.png'):
        required_png(prefix,png)
    rows_csv(prefix,'courbes_precision_rappel.csv')

    rwr=rows_csv('comparaison_rwr_original_symetrise_hepth','resume_comparaison_rwr.csv')
    rwrmap={r['methode']:r for r in rwr}
    if set(rwrmap)!={'Original_u_vers_v','Original_v_vers_u','RWR_symetrise'}:
        raise AssertionError('Comparaison RWR incomplete')
    p1=float(algos['RWR_symetrise']['precision_topk_deterministe'])
    p2=float(rwrmap['RWR_symetrise']['precision_deterministe'])
    if not math.isclose(p1,p2,abs_tol=1e-10):
        raise AssertionError('Precision RWR symetrise differente entre experiences')
    required_png('comparaison_rwr_original_symetrise_hepth','precision_rappel_rwr_comparaison.png')
    rows_csv('comparaison_rwr_original_symetrise_hepth','chevauchement_top_k.csv')

    bench=rows_csv('benchmarks_cn_aa_rwr_hepth','resume_benchmarks.csv')
    names={r['algorithme'] for r in bench}
    if names!={'CN_naif','CN','AA_naif','AA','RWR_symetrise'}:
        raise AssertionError(f'Benchmarks incomplets : {names}')
    sizes={int(r['sommets_graphe']) for r in bench}
    if not sizes: raise AssertionError('Aucune taille de graphe')
    for size in sizes:
        have={r['algorithme'] for r in bench if int(r['sommets_graphe'])==size}
        if have != names: raise AssertionError(f'Algorithmes manquants pour n={size}: {names-have}')
    for png in ('temps_total_echelle_log.png','temps_cn_aa_naif_optimise.png','memoire_supplementaire.png'):
        required_png('benchmarks_cn_aa_rwr_hepth',png)
    rows_csv('benchmarks_cn_aa_rwr_hepth','mesures_individuelles.csv')

    cn=rows_csv('evaluation_cn_aa_hepth','top_1164_cn.csv')
    aa=rows_csv('evaluation_cn_aa_hepth','top_1164_aa.csv')
    r=rows_csv('evaluation_rwr_hepth','top_1164_rwr.csv')
    if any(len(x)!=k for x in (cn,aa,r)):
        raise AssertionError('Top-k incomplet ou discordant')
    for label,data in [('CN',cn),('AA',aa),('RWR_symetrise',r)]:
        n_hits=sum(int(row['relation_future_retrouvee']) for row in data)
        if n_hits != int(algos[label]['tp_topk_deterministe']):
            raise AssertionError(f'Bons liens differents : {label} ({n_hits})')

    title("hep-th | Vérification des sorties")
    print(f"  {n:,} candidats | {k:,} vrais liens | {len(sizes)} tailles de benchmark")
    print("  [OK] CSV et graphiques cohérents")


if __name__=='__main__':
    main()
