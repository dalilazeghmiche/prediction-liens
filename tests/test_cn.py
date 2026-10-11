"""Tests de la mesure CN sur des graphes dont on connaît les réponses.

Lancer depuis la racine du projet :
    python -m unittest discover -s tests -v
"""

import os
import random
import sys
import unittest
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph_io import from_edges
from src.measures.local import common_neighbors, cn_scores, rank

# Graphe de l'exemple fait à la main
EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E")]


class TestCommonNeighbors(unittest.TestCase):
    def setUp(self):
        self.adj = from_edges(EDGES)

    def test_valeurs_a_la_main(self):
        attendus = {("A", "D"): 2, ("B", "E"): 1, ("C", "E"): 1, ("A", "E"): 0}
        for (x, y), cn in attendus.items():
            self.assertEqual(common_neighbors(self.adj, x, y), cn, (x, y))
            self.assertEqual(common_neighbors(self.adj, y, x), cn, "symétrie")

    def test_cn_scores_toutes_paires(self):
        scores = cn_scores(self.adj)
        # A–E (CN = 0) n'apparaît pas ; les paires reliées non plus.
        self.assertEqual(scores, {("A", "D"): 2, ("B", "E"): 1, ("C", "E"): 1})

    def test_classement(self):
        classement = rank(cn_scores(self.adj))
        self.assertEqual(classement[0], (("A", "D"), 2))

    def test_restriction_aux_sommets_coeur(self):
        scores = cn_scores(self.adj, nodes={"A", "B", "C", "D"})
        self.assertEqual(scores, {("A", "D"): 2})

    def test_comparaison_force_brute(self):
        """Sur des graphes aléatoires, la version rapide doit donner
        exactement le même résultat que le test des n² paires."""
        rng = random.Random(42)
        for _ in range(20):
            n = rng.randint(5, 40)
            edges = [(u, v) for u, v in combinations(range(n), 2)
                     if rng.random() < 0.15]
            adj = from_edges(edges)
            attendu = {}
            for x, y in combinations(sorted(adj), 2):
                if y in adj[x]:
                    continue
                c = common_neighbors(adj, x, y)
                if c > 0:
                    attendu[(x, y)] = c
            self.assertEqual(cn_scores(adj), attendu)


if __name__ == "__main__":
    unittest.main()
