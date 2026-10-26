"""Tests de la marche aléatoire avec redémarrage (RWR).

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
from src.measures.global_measures import rwr_from, rwr_matrix, rwr_scores
from src.measures.local import cn_scores

EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("D", "E")]


class TestRwrReference(unittest.TestCase):
    def test_deux_sommets_a_la_main(self):
        """Graphe A–B, α = 0,5. Équations de l'état stationnaire :
           r(A) = α + (1-α)·r(B)   et   r(B) = (1-α)·r(A)
           ⇒ r(A) = 1/(2-α) = 2/3,  r(B) = (1-α)/(2-α) = 1/3."""
        r = rwr_from(from_edges([("A", "B")]), "A", alpha=0.5)
        self.assertAlmostEqual(r["A"], 2 / 3, places=9)
        self.assertAlmostEqual(r["B"], 1 / 3, places=9)

    def test_somme_egale_1(self):
        r = rwr_from(from_edges(EDGES), "A", alpha=0.15)
        self.assertAlmostEqual(sum(r.values()), 1.0, places=9)

    def test_cul_de_sac(self):
        adj = from_edges([("A", "B")], nodes=["Z"])
        r = rwr_from(adj, "Z", alpha=0.3)          # Z n'a aucun voisin
        self.assertAlmostEqual(r["Z"], 1.0)

    def test_redemarrage_fort(self):
        """Plus α est grand, plus le marcheur reste près de sa source."""
        adj = from_edges(EDGES)
        loin = rwr_from(adj, "A", alpha=0.05)["E"]
        pres = rwr_from(adj, "A", alpha=0.8)["E"]
        self.assertGreater(loin, pres)


class TestRwrEfficace(unittest.TestCase):
    def test_meme_resultat_que_la_reference(self):
        rng = random.Random(11)
        for _ in range(10):
            n = rng.randint(5, 30)
            adj = from_edges([(u, v) for u, v in combinations(range(n), 2)
                              if rng.random() < 0.2], nodes=range(n))
            R, nodes, index = rwr_matrix(adj, list(adj), alpha=0.2,
                                         tol=1e-13, block_size=7)
            for j, s in enumerate(adj):
                ref = rwr_from(adj, s, alpha=0.2, tol=1e-13)
                for v in adj:
                    self.assertAlmostEqual(R[index[v], j], ref.get(v, 0.0),
                                           places=8)

    def test_scores_symetrises(self):
        adj = from_edges(EDGES)
        scores = rwr_scores(adj, alpha=0.15)
        ra = rwr_from(adj, "A", alpha=0.15, tol=1e-13)
        re = rwr_from(adj, "E", alpha=0.15, tol=1e-13)
        self.assertAlmostEqual(scores[("A", "E")], ra["E"] + re["A"], places=8)

    def test_voit_plus_loin_que_cn(self):
        """A et E n'ont aucun voisin commun (CN = 0), mais RWR leur donne
        un score > 0 : c'est l'intérêt d'une mesure globale."""
        adj = from_edges(EDGES)
        self.assertNotIn(("A", "E"), cn_scores(adj))
        self.assertGreater(rwr_scores(adj)[("A", "E")], 0)

    def test_pas_de_paire_deja_reliee(self):
        adj = from_edges(EDGES)
        for x, y in rwr_scores(adj):
            self.assertNotIn(y, adj[x])

    def test_composantes_separees(self):
        adj = from_edges([("A", "B"), ("C", "D")])
        self.assertNotIn(("A", "C"), rwr_scores(adj))

    def test_restriction_aux_sommets_coeur(self):
        adj = from_edges(EDGES)
        scores = rwr_scores(adj, nodes={"A", "D", "E"})
        self.assertEqual(set(scores), {("A", "D"), ("A", "E")})


if __name__ == "__main__":
    unittest.main()