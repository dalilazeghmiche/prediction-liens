"""Précision@|test| de CN/AA sur hep-th (1994-96 -> 1997-99)."""
import time

from src.measures.local import cn_scores, aa_scores, rank
from utils_experiences import RESULTS, social_split, save_csv, tie_top_k, title, result, done


def main():
    split = social_split()
    k, candidates = len(split.new_edges), split.num_candidates
    baseline = k / candidates
    output = RESULTS / "evaluation_cn_aa_hepth"
    summary = []
    title("hep-th | CN / AA", info=f"{len(split.core):,} auteurs | {candidates:,} candidats | k={k}")

    for name, method in (("CN", cn_scores), ("AA", aa_scores)):
        start = time.perf_counter()
        ranking = rank(method(split.train_adj, nodes=split.core))
        duration = time.perf_counter() - start
        hits, expected, ties, places = tie_top_k(ranking, split.new_edges, k)
        precision = hits / k

        save_csv(output / f"top_{k}_{name.lower()}.csv",
                 ((i, u, v, repr(score), int((u, v) in split.new_edges))
                  for i, ((u, v), score) in enumerate(ranking[:k], 1)),
                 ["rang", "sommet_u", "sommet_v", "score", "relation_future_retrouvee"])
        summary.append({
            "algorithme": name, "candidats_scores_positifs": len(ranking),
            "k": k, "bonnes_predictions": hits,
            "precision_at_k": f"{precision:.12f}", "rappel_at_k": f"{precision:.12f}",
            "precision_aleatoire": f"{baseline:.12f}",
            "facteur_vs_hasard": f"{precision / baseline:.12f}",
            "ex_aequo_au_seuil": ties, "places_dans_ex_aequo": places,
            "bonnes_predictions_esperees_avec_ex_aequo_aleatoires": f"{expected:.6f}",
            "secondes_calcul_et_tri": f"{duration:.6f}",
        })
        result(name, f"P@{k} = {precision:.2%}", f"{hits} vrais liens", seconds=duration)

    save_csv(output / "resume_cn_aa.csv", summary)
    done(output)


if __name__ == "__main__":
    main()
