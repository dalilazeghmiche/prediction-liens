"""Vérifie les invariants du découpage temporel hep-th."""
from collections import Counter
from itertools import combinations

from src.split import pair_key
from utils_experiences import RAW, social_split, title
from src.datasets import load_hepth


def main():
    papers = load_hepth(RAW / "hep-th-abs.tar.gz", RAW / "hep-th-slacdates.tar.gz")
    split = social_split(papers)
    train, core = split.train_adj, split.core
    train_count, test_count = Counter(), Counter()
    test_edges = set()

    for year, names in papers:
        names = list(dict.fromkeys(names))
        if 1994 <= year <= 1996:
            train_count.update(names)
        elif 1997 <= year <= 1999:
            test_count.update(names)
            test_edges.update(pair_key(u, v) for u, v in combinations(names, 2) if u != v)

    old = sum(v in core for u in core for v in train[u]) // 2
    expected = {pair for pair in test_edges if all(u in core for u in pair)
                and pair[1] not in train[pair[0]]}
    assert all(train_count[u] >= 3 and test_count[u] >= 3 for u in core)
    assert all(u in train.get(v, ()) for v in train for u in train[v])
    assert split.new_edges == expected
    assert split.num_candidates == len(core) * (len(core) - 1) // 2 - old

    title("hep-th | Protocole")
    print(f"  Graphe : {len(train):,} sommets | {sum(map(len, train.values())) // 2:,} arêtes")
    print(f"  Cœur : {len(core):,} auteurs | Test : {len(split.new_edges):,} liens | Candidats : {split.num_candidates:,}")
    print("  [OK] Découpage, cœur et candidats cohérents")


if __name__ == "__main__":
    main()
