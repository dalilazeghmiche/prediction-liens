"""Tests du découpage apprentissage / test.

Lancer depuis la racine du projet :
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph_io import from_edges, num_edges
from src.split import (count_candidate_pairs, papers_from_temporal_edges,
                       random_edge_split, temporal_split)

# Petit jeu d'articles (date, auteurs), calculé à la main :
#   apprentissage 1994-1996 : A:3 articles, B:2, C:1, D:2, E:1
#   test          1997-1999 : A:2, D:2, B:1, C:1, E:1, F:1
PAPERS = [
    (1994, ["A", "B", "C"]),   # arêtes A-B, A-C, B-C
    (1995, ["A", "D"]),
    (1996, ["B", "D"]),
    (1996, ["E", "A"]),
    (1997, ["A", "D"]),        # A-D existait déjà : pas un nouveau lien
    (1998, ["C", "D"]),        # nouveau lien C-D
    (1998, ["B", "E"]),        # nouveau lien B-E
    (1999, ["A", "F"]),        # F absent de l'apprentissage : pas cœur
    (2001, ["C", "E"]),        # hors des deux périodes : ignoré
]
TRAIN, TEST = (1994, 1996), (1997, 1999)


class TestTemporalSplit(unittest.TestCase):
    def test_graphe_apprentissage(self):
        s = temporal_split(PAPERS, TRAIN, TEST, k_train=1, k_test=1)
        self.assertEqual(num_edges(s.train_adj), 6)  # AB AC BC AD BD AE
        self.assertNotIn("F", s.train_adj)          # F n'apparaît qu'au test

    def test_seuils_1(self):
        s = temporal_split(PAPERS, TRAIN, TEST, k_train=1, k_test=1)
        self.assertEqual(s.core, {"A", "B", "C", "D", "E"})
        self.assertEqual(s.new_edges, {("C", "D"), ("B", "E")})

    def test_seuils_2_et_1(self):
        # au moins 2 articles à l'apprentissage : A, B, D
        s = temporal_split(PAPERS, TRAIN, TEST, k_train=2, k_test=1)
        self.assertEqual(s.core, {"A", "B", "D"})
        self.assertEqual(s.new_edges, set())  # A-B, A-D, B-D déjà reliés

    def test_nombre_de_candidats(self):
        s = temporal_split(PAPERS, TRAIN, TEST, k_train=1, k_test=1)
        # 5 sommets cœur → 10 paires, dont 6 déjà reliées → 4 candidates
        self.assertEqual(s.num_candidates, 4)

    def test_auteur_repete_dans_un_article(self):
        s = temporal_split([(1995, ["A", "A", "B"]), (1998, ["A", "B"])],
                           TRAIN, TEST, k_train=1, k_test=1)
        self.assertEqual(num_edges(s.train_adj), 1)
        self.assertNotIn("A", s.train_adj["A"])

    def test_aretes_datees(self):
        papers = papers_from_temporal_edges([("A", "B", 1995), ("B", "C", 1998)])
        self.assertEqual(papers, [(1995, ["A", "B"]), (1998, ["B", "C"])])


class TestRandomSplit(unittest.TestCase):
    def setUp(self):
        self.adj = from_edges([(i, j) for i in range(10) for j in range(i + 1, 10)
                               if (i + j) % 3 == 0])

    def test_proportion_et_disjonction(self):
        s = random_edge_split(self.adj, test_fraction=0.2, seed=1)
        total = num_edges(self.adj)
        self.assertEqual(len(s.new_edges), round(0.2 * total))
        self.assertEqual(num_edges(s.train_adj) + len(s.new_edges), total)
        for u, v in s.new_edges:
            self.assertNotIn(v, s.train_adj[u])     # bien cachée

    def test_reproductible(self):
        a = random_edge_split(self.adj, 0.2, seed=3).new_edges
        b = random_edge_split(self.adj, 0.2, seed=3).new_edges
        self.assertEqual(a, b)

    def test_sommets_conserves(self):
        s = random_edge_split(self.adj, 0.5, seed=0)
        self.assertEqual(set(s.train_adj), set(self.adj))


class TestCountCandidates(unittest.TestCase):
    def test_triangle_plus_un(self):
        adj = from_edges([("A", "B"), ("B", "C"), ("A", "C")], nodes=["D"])
        # 4 sommets → 6 paires, 3 reliées → 3 candidates
        self.assertEqual(count_candidate_pairs(adj, {"A", "B", "C", "D"}), 3)


if __name__ == "__main__":
    unittest.main()