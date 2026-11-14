"""Benchmark de L3 naïf / optimisé sur des sous-graphes induits."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from utils_experiences import title, result, done
import pickle
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent
METHODS = {"L3_naif": "l3_naif", "L3_optimise": "l3_optimise"}


def load_training_graph(data_dir: Path):
    from src.graph_io import from_edges

    nodes_file = data_dir / "nodes.tsv"
    edges_file = data_dir / "train_edges.tsv"
    if not nodes_file.is_file() or not edges_file.is_file():
        raise FileNotFoundError(
            "nodes.tsv ou train_edges.tsv absent. Relancer preparer_donnees_l3.py."
        )
    with nodes_file.open(encoding="utf-8-sig") as fh:
        nodes = {line.strip() for line in fh if line.strip() and not line.startswith("#")}
    edges = []
    with edges_file.open(encoding="utf-8-sig") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"Arête mal formée à la ligne {line_number}")
            u, v = parts[0].strip(), parts[1].strip()
            if u == v or not u or not v:
                raise ValueError(f"Arête invalide à la ligne {line_number}")
            edges.append((u, v))
    if not set(u for edge in edges for u in edge) <= nodes:
        raise ValueError("train_edges.tsv contient un sommet absent de nodes.tsv")
    return from_edges(edges, nodes=nodes)


def nested_induced_subgraph(graph, selected):
    selected = set(selected)
    return {node: graph[node] & selected for node in selected}


def same_scores(naive, optimized):
    if naive.keys() != optimized.keys():
        missing = list(naive.keys() - optimized.keys())[:3]
        extra = list(optimized.keys() - naive.keys())[:3]
        raise AssertionError(f"Paires L3 différentes ; manquantes={missing}, supplémentaires={extra}")
    for pair, value in naive.items():
        if not math.isclose(value, optimized[pair], rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"Score incorrect pour {pair}: {value} != {optimized[pair]}")


def check_random_small_graphs():
    from src.measures.l3 import l3_naif, l3_optimise
    for seed in range(80):
        rng = random.Random(seed)
        nodes = [f"N{i:02d}" for i in range(9)]
        graph = {u: set() for u in nodes}
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if rng.random() < 0.35:
                    graph[u].add(v)
                    graph[v].add(u)
        same_scores(l3_naif(graph), l3_optimise(graph))


def child_worker(args):
    import psutil
    from src.measures.l3 import l3_naif, l3_optimise

    with open(args.payload, "rb") as fh:
        graph = pickle.load(fh)  # Uniquement fichier temporaire local généré par ce script.
    func = l3_naif if args.method == "L3_naif" else l3_optimise
    process = psutil.Process()
    gc.collect()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sampling():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop.wait(0.005)

    thread = threading.Thread(target=sampling, daemon=True)
    thread.start()
    trials = []
    try:
        for _ in range(args.repeats):
            gc.collect()  # non inclus dans le temps
            t0 = time.perf_counter()
            scores = func(graph)
            secs = time.perf_counter() - t0
            trials.append({"seconds": secs, "n_scores": len(scores)})
            time.sleep(0.04)  # permet de capter des allocations persistantes après le calcul
            del scores
    finally:
        stop.set()
        thread.join(timeout=1)
    print(json.dumps({
        "trials": trials,
        "rss_baseline_mib": baseline / 1048576,
        "rss_peak_mib": peak[0] / 1048576,
        "rss_extra_mib": max(0, peak[0] - baseline) / 1048576,
    }), flush=True)


def run_child(method, payload, repeats, timeout):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--payload", str(payload), "--method", method,
           "--repeats", str(repeats)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Erreur de {method} :\n{result.stderr}\n{result.stdout}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def sizes_from_cli(value, max_size):
    items = []
    for x in value.split(","):
        x = x.strip().lower()
        size = max_size if x == "all" else int(x)
        if not 2 <= size <= max_size:
            raise ValueError(f"Taille {size} hors limites (2..{max_size})")
        items.append(size)
    return sorted(set(items))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : CSV créés, graphiques ignorés.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for method in METHODS:
        found = sorted((r for r in summary if r["methode"] == method), key=lambda r: r["n"])
        if found:
            ax.plot([r["n"] for r in found], [r["median_s"] for r in found],
                    marker="o", label=method)
    ax.set_yscale("log")
    ax.set_xlabel("Sommets du sous-graphe induit")
    ax.set_ylabel("Temps médian des scores (s, échelle logarithmique)")
    ax.set_title("HI-II-14 : L3 naïf vs optimisé (temps)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "l3_temps_log.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for method in METHODS:
        found = sorted((r for r in summary if r["methode"] == method), key=lambda r: r["n"])
        if found:
            ax.plot([r["n"] for r in found], [r["rss_extra_mib"] for r in found],
                    marker="o", label=method)
    ax.set_xlabel("Sommets du sous-graphe induit")
    ax.set_ylabel("Mémoire RSS supplémentaire observée (Mio)")
    ax.set_title("HI-II-14 : L3 naïf vs optimisé (mémoire indicative)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "l3_memoire_rss.png", dpi=170)
    plt.close(fig)


def main(args):
    from src.measures.l3 import l3_naif, l3_optimise
    graph = load_training_graph(args.data)
    nodes = sorted(graph)
    rng = random.Random(args.seed)
    rng.shuffle(nodes)
    sizes = sizes_from_cli(args.sizes, len(nodes))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    title("HI-II-14 | Benchmarks L3", info=f"n={sizes} | repeats={args.repeats}")
    check_random_small_graphs()


    summary = []
    individual = []
    gains = []
    with tempfile.TemporaryDirectory(prefix="aaga_l3_") as td:
        payload = Path(td) / "graph.pkl"
        for n in sizes:
            subset = nested_induced_subgraph(graph, nodes[:n])
            edges = sum(map(len, subset.values())) // 2
            if n <= args.naive_max:
                ref = l3_naif(subset)
                fast = l3_optimise(subset)
                same_scores(ref, fast)

                del ref, fast
            with payload.open("wb") as fh:
                pickle.dump(subset, fh, protocol=pickle.HIGHEST_PROTOCOL)
            per_size = {}
            for method in METHODS:
                if method == "L3_naif" and n > args.naive_max:
                    continue
                res = run_child(method, payload, args.repeats, args.worker_timeout)
                timings = [t["seconds"] for t in res["trials"]]
                num = [t["n_scores"] for t in res["trials"]]
                if len(set(num)) != 1:
                    raise AssertionError("Nombre de scores non reproductible")
                record = {
                    "n": n, "m": edges, "methode": method,
                    "scores_positifs": num[0],
                    "median_s": statistics.median(timings),
                    "min_s": min(timings), "max_s": max(timings),
                    "rss_extra_mib": res["rss_extra_mib"],
                    "repetitions": args.repeats,
                }
                summary.append(record)
                per_size[method] = record
                for i, value in enumerate(timings, 1):
                    individual.append({"n": n, "m": edges, "methode": method,
                                       "repetition": i, "temps_s": value})
            if len(per_size) == 2:
                a = per_size["L3_naif"]["median_s"]
                b = per_size["L3_optimise"]["median_s"]
                speedup = a / b if b else float("inf")
                gains.append({"n": n, "m": edges, "acceleration_naif_sur_optimise": speedup})
                result(f"n={n}", f"naïf={a:.4f}s | opt={b:.4f}s | ×{speedup:.1f}")
            else:
                result(f"n={n}", f"opt={per_size['L3_optimise']['median_s']:.4f}s | référence naïve ignorée")

    write_csv(out / "mesures_l3.csv", ["n", "m", "methode", "repetition", "temps_s"], individual)
    write_csv(out / "resume_benchmarks_l3.csv", ["n", "m", "methode", "scores_positifs", "median_s",
                                                  "min_s", "max_s", "rss_extra_mib", "repetitions"], summary)
    write_csv(out / "acceleration_l3.csv", ["n", "m", "acceleration_naif_sur_optimise"], gains)
    protocol = {
        "base": str(args.data), "seed": args.seed, "sizes": sizes,
        "naive_max": args.naive_max, "repetitions": args.repeats,
        "mesure": "temps du calcul de scores seul, sans tri ni préparation des données",
        "sous_graphes": "induits et emboîtés, sélection pseudo-aléatoire de sommets",
        "validation": "égalité des scores sur 80 graphes aléatoires et tailles n <= naive_max",
        "memoire": "delta du pic RSS échantillonné toutes les 5 ms (sous-estimation possible)",
        "interpretation": "la densité change avec n ; étude empirique, pas preuve de complexité asymptotique",
    }
    (out / "protocole_l3.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(summary, out)
    done(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("results/l3_biologique"))
    p.add_argument("--out", type=Path, default=Path("results/l3_biologique/benchmarks"))
    p.add_argument("--sizes", default="500,1000,2000,all")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--naive-max", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--worker-timeout", type=int, default=600)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    p.add_argument("--payload", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.repeats <= 0 or args.naive_max <= 0:
        p.error("--repeats et --naive-max doivent être positifs")
    if args.worker:
        child_worker(args)
    else:
        main(args)
