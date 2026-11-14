"""Précision@|test| du RWR symétrisé sur hep-th."""
import argparse
import time

from src.measures.global_measures import rwr_scores
from src.measures.local import rank
from utils_experiences import (RESULTS, social_split, save_csv, tie_top_k,
                               rwr_options, check_rwr, title, result, done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    rwr_options(parser)
    args = parser.parse_args()
    check_rwr(parser, args)

    split = social_split()
    k, candidates = len(split.new_edges), split.num_candidates
    baseline = k / candidates
    output = RESULTS / "evaluation_rwr_hepth"
    title("hep-th | RWR", info=f"{len(split.core):,} auteurs | {candidates:,} candidats | k={k}")

    start = time.perf_counter()
    ranking = rank(rwr_scores(split.train_adj, nodes=sorted(split.core),
                              alpha=args.alpha, tol=args.tol,
                              max_iter=args.max_iter, block_size=args.block_size))
    duration = time.perf_counter() - start
    hits, expected, tied, places = tie_top_k(ranking, split.new_edges, k)
    precision = hits / k

    save_csv(output / f"top_{k}_rwr.csv",
             ((i, u, v, repr(score), int((u, v) in split.new_edges))
              for i, ((u, v), score) in enumerate(ranking[:k], 1)),
             ["rang", "sommet_u", "sommet_v", "score_rwr_symetrise", "relation_future_retrouvee"])
    save_csv(output / "resume_rwr.csv", [{
        "algorithme": "RWR_symetrise", "alpha": args.alpha, "tol": args.tol,
        "max_iter": args.max_iter, "block_size": args.block_size,
        "auteurs_coeur": len(split.core), "paires_candidates": candidates,
        "nouvelles_relations": k, "paires_scores_positifs": len(ranking),
        "bonnes_predictions": hits, "precision_at_k": precision, "rappel_at_k": precision,
        "hasard": baseline, "facteur_vs_hasard": precision / baseline,
        "ex_aequo_au_seuil": tied, "places_dans_ex_aequo": places,
        "bonnes_predictions_esperees_ex_aequo_aleatoires": expected,
        "temps_calcul_et_tri_secondes": duration,
    }])
    result("RWR", f"P@{k} = {precision:.2%}", f"{hits} vrais liens", seconds=duration)
    done(output)


if __name__ == "__main__":
    main()
