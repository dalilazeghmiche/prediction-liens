"""Compare les dates et statistiques hep-th avec l’article original."""
from collections import Counter
from pathlib import Path
import tarfile

from src.datasets import _read_slacdates, parse_abs
from utils_experiences import RAW, title
from src.split import temporal_split

ABS = RAW / "hep-th-abs.tar.gz"
DATES = RAW / "hep-th-slacdates.tar.gz"


def main():
    if not ABS.is_file() or not DATES.is_file():
        raise SystemExit("Archives introuvables dans data/raw/.")

    date_index = _read_slacdates(DATES)
    papers_slac = []
    papers_submission = []
    count_slac, count_submission = Counter(), Counter()
    missing_dates = missing_authors = cross_period = 0
    n_abs = 0

    with tarfile.open(ABS, "r:gz") as archive:
        for member in archive:
            if not (member.isfile() and member.name.endswith(".abs")):
                continue
            n_abs += 1
            paper_id = Path(member.name).stem
            year_slac = date_index.get(paper_id)
            if year_slac is None:
                missing_dates += 1
            # Le prefixe YYMMNNN identifie l'annee de depot arXiv
            year_submission = 1900 + int(paper_id[:2])
            if year_submission < 1990:
                year_submission += 100
            raw = archive.extractfile(member).read().decode("latin-1")
            authors = parse_abs(raw)
            if not authors:
                missing_authors += 1
                continue

            count_submission[year_submission] += 1
            papers_submission.append((year_submission, authors))
            if year_slac is not None:
                count_slac[year_slac] += 1
                papers_slac.append((year_slac, authors))
                if (1994 <= year_slac <= 1996) != (1994 <= year_submission <= 1996):
                    cross_period += 1

    title("hep-th | Audit du corpus")
    print(f"  {n_abs:,} articles .abs | {missing_dates:,} dates SLAC manquantes | {missing_authors:,} sans auteurs")
    print(f"  {cross_period:,} articles changent de période selon la définition de la date")
    for label, papers in (("Dates SLAC (code actuel)", papers_slac),
                          ("Dates de depot arXiv (comparaison)", papers_submission)):
        s = temporal_split(papers, train=(1994, 1996), test=(1997, 1999),
                           k_train=3, k_test=3)
        train_papers = sum(1 for year, _ in papers if 1994 <= year <= 1996)
        train_edges = sum(len(neigh) for neigh in s.train_adj.values()) // 2
        old_core_edges = sum(1 for u in s.core for v in s.train_adj[u] if v in s.core) // 2
        print(f"  {label}: {train_papers:,} articles train | {len(s.train_adj):,} auteurs | "
              f"{train_edges:,} arêtes | cœur={len(s.core):,} | test={len(s.new_edges):,}")

    print("  Référence article : 9 498 articles train | 5 241 auteurs | cœur=1 438 | test=1 576")
    print("  [INFO] Corpus différent ; scores à comparer uniquement à protocole égal")


if __name__ == "__main__":
    main()
