from pathlib import Path

from src.datasets import load_hepth
from src.split import temporal_split
from src.measures.local import cn_scores, aa_scores, rank
from src.measures.global_measures import rwr_scores

racine = Path(__file__).resolve().parent
abs_path = racine / 'data' / 'raw' / 'hep-th-abs.tar.gz'
dates_path = racine / 'data' / 'raw' / 'hep-th-slacdates.tar.gz'

print('1. Chargement des donnees...')
papers = load_hepth(abs_path, dates_path)
print('Articles charges :', len(papers))

split = temporal_split(papers, train=(1994, 1996), test=(1997, 1999))
graphe = split.train_adj
print('Sommets apprentissage :', len(graphe))
print('Aretes apprentissage :', sum(map(len, graphe.values())) // 2)
print('Sommets du coeur :', len(split.core))
print('Nouveaux liens dans la periode test :', len(split.new_edges))
print('Paires candidates :', split.num_candidates)

print('\n2. CN et Adamic-Adar sur le graphe complet...')
for nom, fonction in [('CN', cn_scores), ('AA', aa_scores)]:
    scores = fonction(graphe, nodes=split.core)
    print(nom, '- paires avec score non nul :', len(scores))
    print('Top 3 :', rank(scores)[:3])

print('\n3. RWR : essai limite a 30 sommets du coeur...')
petit_coeur = sorted(split.core, key=lambda u: (-len(graphe[u]), u))[:30]
scores_rwr = rwr_scores(graphe, nodes=petit_coeur, alpha=0.15, tol=1e-8, max_iter=250, block_size=16)
print('RWR - paires avec score non nul :', len(scores_rwr))
print('Top 3 :', rank(scores_rwr)[:3])
print('\nEssai termine sans erreur.')
