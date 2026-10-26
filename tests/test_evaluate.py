"""Tests de l'évaluation (précision, rappel, hasard).

Lancer depuis la racine du projet :
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import (improvement_factor, precision_at,
                          precision_recall_curve, random_precision)

# Classement de 4 paires ; les 2e et 4e sont de vrais nouveaux liens.
RANKED = [(("a", "b"), 5), (("a", "c"), 4), (("b", "d"), 3), (("c", "d"), 1)]
POSITIVES = {("a", "c"), ("c", "d")}


class TestCourbe(unittest.TestCase):
    def test_valeurs_a_la_main(self):
        ks, p, r = precision_recall_curve(RANKED, POSITIVES)
        self.assertEqual(ks, [1, 2, 3, 4])
        for obtenu, attendu in zip(p, [0, 1 / 2, 1 / 3, 2 / 4]):
            self.assertAlmostEqual(obtenu, attendu)
        for obtenu, attendu in zip(r, [0, 1 / 2, 1 / 2, 1]):
            self.assertAlmostEqual(obtenu, attendu)

    def test_paires_seules_acceptees(self):
        paires = [pair for pair, _ in RANKED]
        self.assertEqual(precision_recall_curve(paires, POSITIVES),
                         precision_recall_curve(RANKED, POSITIVES))

    def test_k_max(self):
        ks, _, _ = precision_recall_curve(RANKED, POSITIVES, k_max=2)
        self.assertEqual(ks, [1, 2])

    def test_sans_lien_a_predire(self):
        with self.assertRaises(ValueError):
            precision_recall_curve(RANKED, set())


class TestPrecisionAtN(unittest.TestCase):
    def test_k_egal_n(self):
        # n = 2 liens à prédire → on regarde les 2 premières : 1 bonne
        self.assertAlmostEqual(precision_at(RANKED, POSITIVES, 2), 0.5)

    def test_classement_trop_court(self):
        # 4 paires classées mais k = 8 : les 4 manquantes comptent faux
        self.assertAlmostEqual(precision_at(RANKED, POSITIVES, 8), 2 / 8)

    def test_hasard_et_facteur(self):
        # 2 liens parmi 100 candidats : le hasard a une précision de 0,02
        self.assertAlmostEqual(random_precision(2, 100), 0.02)
        # précision à n = 2 : 0,5  →  facteur 0,5 / 0,02 = 25
        self.assertAlmostEqual(improvement_factor(RANKED, POSITIVES, 100), 25)


class TestBoutEnBout(unittest.TestCase):
    """Chaîne complète : découpage → CN → classement → évaluation."""

    def test_cn_sur_petit_jeu(self):
        from src.measures.local import cn_scores, rank
        from src.split import temporal_split
        papers = [(1994, ["A", "B", "C"]), (1995, ["A", "D"]),
                  (1996, ["B", "D"]), (1996, ["E", "A"]),
                  (1998, ["C", "D"]), (1998, ["B", "E"])]
        s = temporal_split(papers, (1994, 1996), (1997, 1999),
                           k_train=1, k_test=1)
        ranked = rank(cn_scores(s.train_adj, nodes=s.core))
        # CN : C-D = 2 (A, B), B-E = 1 (A), C-E = 1 (A), D-E = 1 (A)
        self.assertEqual(ranked[0], (("C", "D"), 2))
        ks, p, r = precision_recall_curve(ranked, s.new_edges)
        self.assertEqual(r[-1], 1.0)     # les deux liens sont retrouvés


if __name__ == "__main__":
    unittest.main()