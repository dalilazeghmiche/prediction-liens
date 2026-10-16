"""Tests de la mesure Adamic-Adar (AA).

Lancer depuis la racine du projet :
    python -m unittest discover -s tests -v
"""

import math
import os
import random
import sys
import unittest
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph_io import from_edges
from src.measures.local import adamic_adar, aa_scores, cn_scores, rank

# Même graphe que pour CN :
# A–B, A–C, B–C, B–D, C–D, D–E   →  d(B) = d(C) = d(D) = 3
EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E")]


class TestAdamicAdar(unittest.TestCase):
    def setUp(self):
        self.adj = from_edges(EDGES)

    def test_valeurs_a_la_main(self):
        un_sur_ln3 = 1 / math.log(3)
        attendus = {
            ("A", "D"): 2 * un_sur_ln3,   # voisins communs B et C (degré 3)
            ("B", "E"): un_sur_ln3,       # voisin commun D (degré 3)
            ("C", "E"): un_sur_ln3,       # voisin commun D (degré 3)
            ("A", "E"): 0.0,              # aucun voisin commun
        }
        for (x, y), aa in attendus.items():
            self.assertAlmostEqual(adamic_adar(self.adj, x, y), aa, places=12)
            self.assertAlmostEqual(adamic_adar(self.adj, y, x), aa, places=12)

    def test_aa_scores_toutes_paires(self):
        scores = aa_scores(self.adj)
        self.assertEqual(set(scores), {("A", "D"), ("B", "E"), ("C", "E")})
        self.assertAlmostEqual(scores[("A", "D")], 2 / math.log(3), places=12)

    def test_memes_paires_que_cn(self):
        """AA et CN considèrent exactement les mêmes paires candidates."""
        self.assertEqual(set(aa_scores(self.adj)), set(cn_scores(self.adj)))

    def test_hub_compte_moins(self):
        """Ce qui distingue AA de CN : un voisin commun très connecté
        compte moins qu'un voisin commun peu connecté.
        - x et y partagent w, de degré 2      → AA = 1/ln 2 ≈ 1,44
        - a et b partagent le hub H, degré 5  → AA = 1/ln 5 ≈ 0,62
        Les deux paires ont pourtant le même CN (= 1)."""
        adj = from_edges([("x", "w"), ("w", "y"),
                          ("H", "a"), ("H", "b"), ("H", "c"),
                          ("H", "d"), ("H", "e")])
        cn = cn_scores(adj)
        self.assertEqual(cn[("x", "y")], cn[("a", "b")])
        aa = aa_scores(adj)
        self.assertAlmostEqual(aa[("x", "y")], 1 / math.log(2), places=12)
        self.assertAlmostEqual(aa[("a", "b")], 1 / math.log(5), places=12)
        self.assertEqual(rank(aa)[0][0], ("x", "y"))

    def test_restriction_aux_sommets_coeur(self):
        # Seule A–D a ses deux extrémités dans le cœur ; d(B) et d(C)
        # restent calculés sur tout le graphe (degré 3).
        scores = aa_scores(self.adj, nodes={"A", "B", "C", "D"})
        self.assertEqual(set(scores), {("A", "D")})
        self.assertAlmostEqual(scores[("A", "D")], 2 / math.log(3), places=12)

    def test_comparaison_force_brute(self):
        rng = random.Random(7)
        for _ in range(20):
            n = rng.randint(5, 40)
            edges = [(u, v) for u, v in combinations(range(n), 2)
                     if rng.random() < 0.15]
            adj = from_edges(edges)
            attendu = {}
            for x, y in combinations(sorted(adj), 2):
                if y in adj[x]:
                    continue
                s = adamic_adar(adj, x, y)
                if s > 0:
                    attendu[(x, y)] = s
            obtenu = aa_scores(adj)
            self.assertEqual(set(obtenu), set(attendu))
            for k in attendu:
                self.assertAlmostEqual(obtenu[k], attendu[k], places=9)


if __name__ == "__main__":
    unittest.main()