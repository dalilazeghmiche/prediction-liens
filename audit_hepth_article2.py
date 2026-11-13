"""Audit des donnees hep-th du projet AAGA (article [2]).

"""
from collections import Counter
from pathlib import Path
import tarfile

from src.datasets import _read_slacdates, parse_abs
from src.split import temporal_split

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
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

    print("\n=== AUDIT DES ARCHIVES hep-th ===")
    print(f"Fichiers .abs dans l'archive : {n_abs:,}")
    print(f"Identifiants dans les dates SLAC : {len(date_index):,}")
    print(f"Articles sans date SLAC : {missing_dates:,}")
    print(f"Articles sans auteurs reconnus : {missing_authors:,}")

    print("\n=== ARTICLES PAR ANNEE (deux definitions de la date) ===")
    print(f"{'Annee':>6} | {'SLAC':>8} | {'Depot arXiv':>12}")
    print("-" * 34)
    for year in range(1993, 2000):
        print(f"{year:>6} | {count_slac[year]:>8,} | {count_submission[year]:>12,}")
    print(f"{'1994-96':>6} | {sum(count_slac[y] for y in range(1994,1997)):>8,} | {sum(count_submission[y] for y in range(1994,1997)):>12,}")
    print(f"Articles traversant la frontiere 1994-96 selon la date choisie : {cross_period:,}")

    print("\n=== RESULTATS AVEC LE PROTOCOLE DU PROJET ===")
    print("Apprentissage : 1994-1996 ; test : 1997-1999 ; seuils : 3 et 3")
    for label, papers in (("Dates SLAC (code actuel)", papers_slac),
                          ("Dates de depot arXiv (comparaison)", papers_submission)):
        s = temporal_split(papers, train=(1994, 1996), test=(1997, 1999),
                           k_train=3, k_test=3)
        train_papers = sum(1 for year, _ in papers if 1994 <= year <= 1996)
        train_edges = sum(len(neigh) for neigh in s.train_adj.values()) // 2
        old_core_edges = sum(1 for u in s.core for v in s.train_adj[u] if v in s.core) // 2
        print(f"\n{label} :")
        print(f"  Publications d'apprentissage : {train_papers:,}")
        print(f"  Auteurs d'apprentissage       : {len(s.train_adj):,}")
        print(f"  Aretes distinctes non orientees : {train_edges:,}")
        print(f"  Auteurs du coeur             : {len(s.core):,}")
        print(f"  Anciennes aretes dans le coeur : {old_core_edges:,}")
        print(f"  Nouvelles aretes du test     : {len(s.new_edges):,}")
        print(f"  Paires candidates            : {s.num_candidates:,}")

    print("\n=== REFERENCE DE L'ARTICLE [2] (hep-th) ===")
    print("Publications d'apprentissage : 9 498 ; auteurs : 5 241 ; coeur : 1 438 ;")
    print("nouvelles collaborations dans le coeur : 1 576.")
    print("ATTENTION : les auteurs comptent certaines collaborations comme")
    print("paires ordonnees / interactions ; ne pas comparer directement")
    print("leur colonne 'collaborations' aux aretes distinctes non orientees.")
    print("\nBILAN : ce script diagnostique les ecarts. Il ne prouve pas")
    print("que les fichiers utilises par l'article sont strictement identiques.")


if __name__ == "__main__":
    main()
