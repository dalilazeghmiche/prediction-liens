"""Outils partagés par les scripts d'expériences (pas par les algorithmes)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"


def social_split(papers=None):
    """Charge le même protocole hep-th pour toutes les expériences sociales."""
    from src.datasets import load_hepth
    from src.split import temporal_split

    files = [RAW / "hep-th-abs.tar.gz", RAW / "hep-th-slacdates.tar.gz"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Donnée manquante : {path}")
    return temporal_split(load_hepth(*files) if papers is None else papers, train=(1994, 1996),
                          test=(1997, 1999), k_train=3, k_test=3)


def save_csv(path, rows, columns=None, *, bom=False):
    """Enregistre des dictionnaires ou des lignes, avec l'en-tête fourni."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
        if columns is None:
            rows = list(rows)
            if not rows:
                raise ValueError("Impossible d'écrire un CSV vide sans colonnes")
            columns = list(rows[0])
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)


def title(name, *, info=None):
    print(f"\n[{name}]" + (f" {info}" if info else ""), flush=True)


def result(name, score, hits=None, *, seconds=None, detail=None):
    line = f"  {name:<16} {score}"
    if hits is not None:
        line += f" | {hits}"
    if seconds is not None:
        line += f" | {seconds:.2f} s"
    if detail:
        line += f" | {detail}"
    print(line, flush=True)


def done(path):
    try:
        display = Path(path).relative_to(ROOT)
    except ValueError:
        display = path
    print(f"  [OK] Résultats : {display}/", flush=True)


def rwr_options(parser):
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=64)


def check_rwr(parser, args):
    if not (0 < args.alpha < 1 and args.tol > 0 and
            args.max_iter > 0 and args.block_size > 0):
        parser.error("RWR : 0 < alpha < 1, tol > 0, max-iter et block-size > 0")


def tie_top_k(ranked, positives, k):
    """Top-k déterministe et espérance uniforme des ex æquo au seuil (scores >0)."""
    if len(ranked) < k:
        raise ValueError("Moins de k scores positifs : traiter les scores nuls")
    hits = sum(pair in positives for pair, _ in ranked[:k])
    threshold = ranked[k - 1][1]
    above = tied = good_above = good_tied = 0
    for pair, score in ranked:
        if score > threshold:
            above += 1
            good_above += pair in positives
        elif score == threshold:
            tied += 1
            good_tied += pair in positives
        else:
            break
    places = k - above
    expected = good_above + places * good_tied / tied
    return hits, expected, tied, places

# Calcul exact des ex æquo, y compris le groupe implicite de scores nuls.
from bisect import bisect_left
from dataclasses import dataclass
import time

@dataclass
class RankingStats:
    name: str
    num_candidates: int
    num_positives: int
    positive_scores: int
    positives_positive_scores: int
    elapsed: float
    ends: list[int]
    cumulative_hits: list[int]
    group_sizes: list[int]
    group_hits: list[int]
    deterministic_hits_at_k: int
    tie_size_at_k: int
    tie_slots_at_k: int

    def expected_hits(self, k: int) -> float:
        """Espérance exacte des TP au top-k, ex æquo départagés au hasard."""
        if not (1 <= k <= self.num_candidates):
            raise ValueError("k doit être entre 1 et le nombre de candidats")
        i = bisect_left(self.ends, k)
        previous_end = 0 if i == 0 else self.ends[i - 1]
        previous_hits = 0 if i == 0 else self.cumulative_hits[i - 1]
        return previous_hits + (k - previous_end) * self.group_hits[i] / self.group_sizes[i]

    def expected_precision_recall(self, k: int) -> tuple[float, float]:
        hits = self.expected_hits(k)
        return hits / k, hits / self.num_positives


def analyse_ranking(name, scores, positives, num_candidates, k_reference, started_at):
    """Traite seulement les scores > 0 et représente le reste par un bloc nul."""
    if len(scores) > num_candidates:
        raise AssertionError("Plus de scores que de paires candidates")
    if any(score <= 0 for score in scores.values()):
        raise AssertionError("Les scores retournés doivent être strictement positifs")

    from src.measures.local import rank
    ranked = rank(scores)  # même règle déterministe que le projet
    deterministic_hits = sum(pair in positives for pair, _ in ranked[:k_reference])

    ends, cumulative_hits, group_sizes, group_hits = [], [], [], []
    last_score = None
    group_size, group_pos = 0, 0
    total_count, total_pos = 0, 0

    def emit_group():
        nonlocal total_count, total_pos
        if group_size:
            total_count += group_size
            total_pos += group_pos
            ends.append(total_count)
            cumulative_hits.append(total_pos)
            group_sizes.append(group_size)
            group_hits.append(group_pos)

    for pair, score in ranked:
        if last_score is not None and score != last_score:
            emit_group()
            group_size, group_pos = 0, 0
        last_score = score
        group_size += 1
        group_pos += int(pair in positives)
    emit_group()

    if total_count != len(scores):
        raise AssertionError("Erreur interne dans les groupes ex æquo")

    # Compléter avec toutes les paires non produites par l'algorithme (score 0).
    zeros = num_candidates - len(scores)
    positives_in_zeros = len(positives) - total_pos
    if not 0 <= positives_in_zeros <= zeros:
        raise AssertionError("Les positifs dans le groupe zéro sont incohérents")
    if zeros:
        ends.append(num_candidates)
        cumulative_hits.append(len(positives))
        group_sizes.append(zeros)
        group_hits.append(positives_in_zeros)

    if ends[-1] != num_candidates or cumulative_hits[-1] != len(positives):
        raise AssertionError("Tous les candidats et vrais liens doivent être représentés")

    idx = bisect_left(ends, k_reference)
    above = ends[idx - 1] if idx else 0
    stats = RankingStats(
        name=name,
        num_candidates=num_candidates,
        num_positives=len(positives),
        positive_scores=len(scores),
        positives_positive_scores=total_pos,
        elapsed=time.perf_counter() - started_at,
        ends=ends,
        cumulative_hits=cumulative_hits,
        group_sizes=group_sizes,
        group_hits=group_hits,
        deterministic_hits_at_k=deterministic_hits,
        tie_size_at_k=group_sizes[idx],
        tie_slots_at_k=k_reference - above,
    )
    return stats


def selection_ks(total, k_ref):
    """Densité fine au début ; échantillonnage progressif vers tous les candidats."""
    ks = set(range(1, min(2500, total) + 1))
    ks.update(range(2501, min(total, 10000) + 1, 10))
    ks.update(range(10001, min(total, 100000) + 1, 250))
    ks.update(range(100001, total + 1, 2000))
    ks.update({total, k_ref, min(100, total), min(500, total), min(1000, total)})
    return sorted(k for k in ks if 1 <= k <= total)


from collections import defaultdict
import math

def courbe_attendue(scores, vrais_liens, total_candidats, max_k):
    """Courbe de précision/rappel en espérance avec tirage uniforme des ex æquo.

    Une paire de score absent est de score zéro. O(P+G log G+max_k)
    plutôt que O(nombre_total_de_candidats).
    """
    if max_k <= 0 or total_candidats <= 0 or not vrais_liens:
        raise ValueError("Effectifs invalides")
    if len(scores) > total_candidats:
        raise ValueError("Davantage de scores que de candidats")

    groupes = defaultdict(lambda: [0, 0])
    hits_positifs = 0
    for pair, score in scores.items():
        if not math.isfinite(score) or score <= 0:
            raise ValueError(f"Score non positif ou non fini : {pair}: {score}")
        hit = pair in vrais_liens
        groupes[score][0] += 1
        groupes[score][1] += int(hit)
        hits_positifs += hit

    nb_zero = total_candidats - len(scores)
    hits_zero = len(vrais_liens) - hits_positifs
    if not 0 <= hits_zero <= nb_zero:
        raise ValueError("Incohérence dans les liens de test et les scores")
    if nb_zero:
        groupes[0.0] = [nb_zero, hits_zero]

    k_cap = min(max_k, total_candidats)
    rows = []
    k = 0
    hits_cumules = 0.0
    for score in sorted(groupes, reverse=True):
        nombre, bons = groupes[score]
        for j in range(1, min(nombre, k_cap - k) + 1):
            kk = k + j
            hits_moyens = hits_cumules + j * bons / nombre
            rows.append((kk, hits_moyens / kk, hits_moyens / len(vrais_liens)))
        k += nombre
        hits_cumules += bons
        if k >= k_cap:
            break
    return rows, hits_positifs

