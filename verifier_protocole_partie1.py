"""Vérifie le découpage temporel de l'article [2] sur hep-th.

"""
from collections import Counter
from itertools import combinations
from pathlib import Path

from src.datasets import load_hepth
from src.split import temporal_split, pair_key

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

papers = load_hepth(RAW / "hep-th-abs.tar.gz", RAW / "hep-th-slacdates.tar.gz")
s = temporal_split(papers, train=(1994, 1996), test=(1997, 1999), k_train=3, k_test=3)

train_counts, test_counts = Counter(), Counter()
test_pairs = set()
for year, authors in papers:
    authors = list(dict.fromkeys(authors))
    if 1994 <= year <= 1996:
        train_counts.update(authors)
    elif 1997 <= year <= 1999:
        test_counts.update(authors)
        test_pairs.update(pair_key(a, b) for a, b in combinations(authors, 2) if a != b)

train = s.train_adj
core = s.core
expected_new = {p for p in test_pairs if p[0] in core and p[1] in core and p[1] not in train[p[0]]}
edges_core = sum(1 for u in core for v in train[u] if v in core) // 2
expected_candidates = len(core) * (len(core) - 1) // 2 - edges_core

assert all(train_counts[u] >= 3 and test_counts[u] >= 3 for u in core)
assert all(u in train for u in core)
assert all(u != v and u in train[v] for v in train for u in train[v])
assert s.new_edges == expected_new
assert s.num_candidates == expected_candidates
assert s.new_edges.isdisjoint({pair_key(u, v) for u in train for v in train[u]})

print("PROTOCOLE ARTICLE [2] : hep-th")
print("Entrainement : 1994–1996 ; test : 1997–1999")
print("Seuil d'activite : >= 3 articles dans chaque periode")
print(f"Articles charges (toutes dates) : {len(papers):,}")
print(f"Auteurs du graphe d'entrainement : {len(train):,}")
print(f"Aretes du graphe d'entrainement : {sum(map(len, train.values())) // 2:,}")
print(f"Auteurs du coeur : {len(core):,}")
print(f"Anciennes aretes entre auteurs du coeur : {edges_core:,}")
print(f"Nouvelles collaborations a retrouver : {len(s.new_edges):,}")
print(f"Paires candidates : {s.num_candidates:,}")
print("OK : criteres du coeur, symetrie, nouveaux liens et candidats verifies.")
