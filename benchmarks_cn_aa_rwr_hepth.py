"""Benchmarks CN/AA naïfs et optimisés, RWR : temps et mémoire RSS."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import pickle
from pathlib import Path
from utils_experiences import title, result, done, social_split
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "benchmarks_cn_aa_rwr_hepth"
ALGORITHMS = ["CN_naif", "CN", "AA_naif", "AA", "RWR_symetrise"]


def cn_naif(adj, core):
    """Référence indépendante : toutes les paires du coeur, intersections."""
    core = sorted(core)
    scores = {}
    for i, u in enumerate(core):
        for v in core[i + 1:]:
            if v in adj[u]:
                continue
            score = len(adj[u] & adj[v])
            if score:
                scores[(u, v)] = score
    return scores


def aa_naif(adj, core):
    """Référence indépendante : toutes les paires du coeur, intersections."""
    core = sorted(core)
    scores = {}
    for i, u in enumerate(core):
        for v in core[i + 1:]:
            if v in adj[u]:
                continue
            common = adj[u] & adj[v]
            if common:
                scores[(u, v)] = sum(1.0 / math.log(len(adj[w])) for w in common)
    return scores


def do_scores(algo, adj, core, args):
    from src.measures.local import cn_scores, aa_scores
    from src.measures.global_measures import rwr_scores

    if algo == "CN_naif":
        return cn_naif(adj, core)
    if algo == "CN":
        return cn_scores(adj, nodes=core)
    if algo == "AA_naif":
        return aa_naif(adj, core)
    if algo == "AA":
        return aa_scores(adj, nodes=core)
    if algo == "RWR_symetrise":
        return rwr_scores(adj, nodes=sorted(core), alpha=args.alpha,
                          tol=args.tol, max_iter=args.max_iter,
                          block_size=args.block_size)
    raise ValueError(f"Algorithme inconnu : {algo}")


def worker(args):
    """Répète les calculs dans un processus dédié, avec sondage RSS interne."""
    import psutil
    import threading
    from src.measures.local import rank
    from src.measures.global_measures import rwr_scores  # précharge SciPy

    with open(args.payload, "rb") as f:
        data = pickle.load(f)  # fichier temporaire créé localement
    adj, core = data["adj"], data["core"]
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop.wait(0.005)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    trials = []
    try:
        for _ in range(args.repeats):
            gc.collect()  # hors chronométrage
            t0 = time.perf_counter()
            scores = do_scores(args.algo, adj, core, args)
            score_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            ranked = rank(scores)
            sort_time = time.perf_counter() - t1
            trials.append({"score_seconds": score_time,
                           "sort_seconds": sort_time,
                           "scores": len(scores)})
            # Laisser au moniteur le temps d'observer les objets produits.
            time.sleep(0.06)
            del scores, ranked
    finally:
        stop.set()
        watcher.join(timeout=2)

    print(json.dumps({"event": "done", "trials": trials,
                      "rss_peak_mb": peak[0] / (1024 * 1024),
                      "rss_baseline_mb": baseline / (1024 * 1024),
                      "rss_extra_peak_mb": max(0, peak[0] - baseline) / (1024 * 1024)}),
          flush=True)


def measure_process(algo, payload_path, args):
    """Chaque méthode et taille dispose d'un processus isolé (N répétitions)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--payload", str(payload_path), "--algo", algo,
           "--repeats", str(args.repeats),
           "--alpha", str(args.alpha), "--tol", str(args.tol),
           "--max-iter", str(args.max_iter), "--block-size", str(args.block_size)]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=args.worker_timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"{algo} a dépassé {args.worker_timeout} s ; "
                           "augmente --worker-timeout ou réduis les tailles") from e
    if proc.returncode:
        raise RuntimeError(f"{algo}: échec du processus :\n{proc.stderr}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{algo}: aucun résultat : {proc.stderr}")
    result = json.loads(lines[-1])
    if result.get("event") != "done" or len(result["trials"]) != args.repeats:
        raise RuntimeError(f"Sortie inattendue pour {algo}: {result}")
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_sizes(spec, maximum):
    sizes = []
    for item in spec.split(","):
        part = item.strip().lower()
        if part == "all":
            size = maximum
        else:
            try:
                size = int(part)
            except ValueError:
                raise ValueError(f"Taille incorrecte : {part}")
        if not 2 <= size <= maximum:
            raise ValueError(f"Taille {size} hors de [2, {maximum}]")
        sizes.append(size)
    return sorted(set(sizes))


def verify_local_algorithms(adj, core):
    """Contrôle hors chronométrage : même scores pour naïf et optimisé."""
    from src.measures.local import cn_scores, aa_scores
    ref_cn = cn_naif(adj, core)
    got_cn = cn_scores(adj, nodes=core)
    if ref_cn != got_cn:
        raise AssertionError("Les scores CN naïfs et optimisés diffèrent")
    ref_aa = aa_naif(adj, core)
    got_aa = aa_scores(adj, nodes=core)
    if ref_aa.keys() != got_aa.keys() or any(
        not math.isclose(ref_aa[k], got_aa[k], rel_tol=1e-12, abs_tol=1e-12)
        for k in ref_aa
    ):
        raise AssertionError("Les scores AA naïfs et optimisés diffèrent")


def make_plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(metric, algos, filename, ylabel, log=False):
        fig, ax = plt.subplots(figsize=(9, 5.6))
        for algo in algos:
            rows = sorted((r for r in summary if r["algorithme"] == algo),
                          key=lambda r: r["sommets_graphe"])
            ax.plot([r["sommets_graphe"] for r in rows],
                    [r[metric] for r in rows], marker="o", label=algo)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Nombre de sommets du sous-graphe induit")
        ax.set_ylabel(ylabel)
        ax.set_title("hep-th — benchmarks sur sous-graphes induits")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=170)
        plt.close(fig)

    curve("mediane_total_s", ALGORITHMS,
          "temps_total_echelle_log.png", "Temps calcul + tri, médiane (s)", log=True)
    curve("mediane_total_s", ["CN_naif", "CN", "AA_naif", "AA"],
          "temps_cn_aa_naif_optimise.png", "Temps calcul + tri, médiane (s)")
    curve("mediane_rss_extra_mb", ALGORITHMS,
          "memoire_supplementaire.png", "RSS supplémentaire max. échantillonné, médiane (Mio)")


def main(args):
    from src.datasets import load_hepth
    from src.split import temporal_split

    if args.repeats < 1:
        raise ValueError("--repeats doit être supérieur ou égal à 1")
    if not (0 < args.alpha < 1 and args.tol > 0 and args.block_size > 0 and args.max_iter > 0):
        raise ValueError("Paramètres RWR invalides")
    a = ROOT / "data" / "raw" / "hep-th-abs.tar.gz"
    d = ROOT / "data" / "raw" / "hep-th-slacdates.tar.gz"
    if not (a.exists() and d.exists()):
        raise FileNotFoundError("Archives hep-th introuvables dans data/raw/")
    split = social_split()
    adj_full = split.train_adj
    all_nodes = sorted(adj_full)
    rng = random.Random(args.seed)
    rng.shuffle(all_nodes)
    sizes = parse_sizes(args.sizes, len(all_nodes))
    OUT.mkdir(parents=True, exist_ok=True)
    title("hep-th | Benchmarks", info=f"n={sizes} | repeats={args.repeats}")

    details = []
    detail_cols = ["sommets_graphe", "aretes_graphe", "sommets_coeur",
                   "paires_candidates", "algorithme", "repetition",
                   "nb_scores_positifs", "temps_scores_s", "temps_tri_s",
                   "temps_total_s", "rss_base_mib", "rss_pic_mib",
                   "rss_supplementaire_pic_mib"]
    with tempfile.TemporaryDirectory(prefix="hepth_benchmark_") as tmpdir:
        for n in sizes:
            selected = set(all_nodes[:n])
            core = sorted(selected.intersection(split.core))
            adj = {u: adj_full[u].intersection(selected) for u in sorted(selected)}
            edges = sum(len(v) for v in adj.values()) // 2
            candidates = len(core) * (len(core) - 1) // 2 - sum(
                1 for u in core for v in adj[u] if v in core) // 2
            payload_file = Path(tmpdir) / f"graph_{n}.pickle"
            with payload_file.open("wb") as f:
                pickle.dump({"adj": adj, "core": core}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            verify_local_algorithms(adj, core)
            counts = {}
            for algo in ALGORITHMS:
                r = measure_process(algo, payload_file, args)
                ncounts = {t["scores"] for t in r["trials"]}
                if len(ncounts) != 1:
                    raise AssertionError(f"Nombre de scores instable pour {algo} (n={n})")
                counts[algo] = ncounts.pop()
                for rep, trial in enumerate(r["trials"], start=1):
                    details.append({
                        "sommets_graphe": n, "aretes_graphe": edges,
                        "sommets_coeur": len(core), "paires_candidates": candidates,
                        "algorithme": algo, "repetition": rep,
                        "nb_scores_positifs": trial["scores"],
                        "temps_scores_s": trial["score_seconds"],
                        "temps_tri_s": trial["sort_seconds"],
                        "temps_total_s": trial["score_seconds"] + trial["sort_seconds"],
                        "rss_base_mib": r["rss_baseline_mb"],
                        "rss_pic_mib": r["rss_peak_mb"],
                        "rss_supplementaire_pic_mib": r["rss_extra_peak_mb"],
                    })
                median_run = statistics.median(
                    t["score_seconds"] + t["sort_seconds"] for t in r["trials"])
                result(f"n={n} {algo}", f"{median_run:.4f} s",
                       detail=f"RSS +{r['rss_extra_peak_mb']:.1f} Mio")
                # Sauvegarde immédiate en cas d'interruption d'un benchmark long.
                write_csv(OUT / "mesures_individuelles.csv", details, detail_cols)
            if counts["CN"] != counts["CN_naif"]:
                raise AssertionError(f"CN optimisé/naïf : nombres de scores différents (n={n})")
            if counts["AA"] != counts["AA_naif"]:
                raise AssertionError(f"AA optimisé/naïf : nombres de scores différents (n={n})")

    write_csv(OUT / "mesures_individuelles.csv", details, detail_cols)
    summary = []
    for n in sizes:
        for algo in ALGORITHMS:
            group = [r for r in details if r["sommets_graphe"] == n and
                     r["algorithme"] == algo]
            all_times = [r["temps_total_s"] for r in group]
            s = group[0]
            summary.append({
                "sommets_graphe": n, "aretes_graphe": s["aretes_graphe"],
                "sommets_coeur": s["sommets_coeur"],
                "paires_candidates": s["paires_candidates"],
                "algorithme": algo, "repetitions": len(group),
                "nb_scores_positifs": s["nb_scores_positifs"],
                "mediane_scores_s": statistics.median(r["temps_scores_s"] for r in group),
                "mediane_tri_s": statistics.median(r["temps_tri_s"] for r in group),
                "mediane_total_s": statistics.median(all_times),
                "min_total_s": min(all_times), "max_total_s": max(all_times),
                "mediane_rss_extra_mb": statistics.median(
                    r["rss_supplementaire_pic_mib"] for r in group),
                "mediane_rss_peak_mb": statistics.median(r["rss_pic_mib"] for r in group),
            })
    fields = ["sommets_graphe", "aretes_graphe", "sommets_coeur", "paires_candidates",
              "algorithme", "repetitions", "nb_scores_positifs", "mediane_scores_s",
              "mediane_tri_s", "mediane_total_s", "min_total_s", "max_total_s",
              "mediane_rss_extra_mb", "mediane_rss_peak_mb"]
    write_csv(OUT / "resume_benchmarks.csv", summary, fields)
    make_plots(summary)
    txt = f"""# Protocole des benchmarks (article [2] — hep-th)

- Source : hep-th KDD Cup 2003, années SLAC.
- Apprentissage : 1994–1996 ; cœur historique : >=3 articles en train ET en test 1997–1999.
- Sous-graphes : **induits**, échantillonnage pseudo-aléatoire avec graine {args.seed}, sous-ensembles emboîtés.
- Tailles (sommets) : {sizes} ; RÉPÉTITIONS : {args.repeats} par taille et méthode.
- Pour chaque taille : les cinq méthodes utilisent exactement les mêmes sommets,
  arêtes et sommets candidats du cœur restreint.
- Mesures : CN/AA optimisés du dépôt, CN/AA naïfs de ce script (références),
  RWR symétrisé du dépôt ; tous calculent les scores **et** trient le résultat.
- RWR : alpha={args.alpha}, tol={args.tol}, max_iter={args.max_iter}, block_size={args.block_size}.
- Chronométrage : `time.perf_counter()` dans un processus enfant par méthode et taille,
  avec plusieurs répétitions successives. Bibliothèques BLAS fixées à 1 thread.
  Import des bibliothèques et chargement des données exclus du chronométrage.
- Mémoire : pic RSS du processus enfant, **échantillonné** toutes les ~5 ms
  dans un thread indépendant. Le pic est commun aux répétitions du même processus.
  La colonne supplémentaire est max(0, pic RSS - RSS juste avant calcul).
  Le pic réel peut être plus élevé ; l'exactitude est limitée pour les étapes très rapides.
- Résumés : médiane, minimum et maximum des temps ; médiane des pics RSS observés.
- La génération des CSV/figures n'est pas comprise dans les mesures.
- Les tailles croissantes servent aux BENCHMARKS ALGORITHMIQUES.
  Les sous-graphes ne reproduisent pas l'expérience temporelle de prédiction de liens.
- Les scores RWR sont symétrisés : r_u(v) + r_v(u), adaptation de l'article [2].
- CN/AA naïfs et optimisés : égalité des scores vérifiée sur chaque
  sous-graphe (AA à une tolérance flottante de 1e-12), hors chronométrage.
- Sur de très grands graphes, RWR peut consommer beaucoup de mémoire car la
  fonction du dépôt construit R (n × |coeur|) et S (|coeur| × |coeur|) denses.
- Le corpus de ce dépôt diffère du corpus exact de l'article [2].
"""
    (OUT / "protocole_benchmarks.md").write_text(txt, encoding="utf-8")
    done(OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmarks CN/AA/RWR sur hep-th, sans modifier src/")
    p.add_argument("--sizes", default="500,1000,2000,all",
                   help="Tailles des sous-graphes, ex. 500,1000,2000,all")
    p.add_argument("--repeats", type=int, default=3, help="Nombre de répétitions (défaut 3)")
    p.add_argument("--seed", type=int, default=42, help="Graine de l'échantillonnage (défaut 42)")
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--worker-timeout", type=float, default=120.0,
                   help="Temps limite par essai en secondes (défaut 120)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--payload", help=argparse.SUPPRESS)
    p.add_argument("--algo", help=argparse.SUPPRESS)
    ar = p.parse_args()
    if ar.worker:
        worker(ar)
    else:
        main(ar)
