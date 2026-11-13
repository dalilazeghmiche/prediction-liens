"""Tests du traitement des ex æquo sans énumérer les candidats."""
import unittest
from evaluer_l3_bio import courbe_attendue, top_k_deterministe


class TestEvaluationL3Bio(unittest.TestCase):
    def test_ex_aequo_et_zero(self):
        # 6 candidats, dont 2 vrais liens. Scores connus pour trois paires.
        # Deux candidats au score 1, dont un vrai. Trois scores nuls,
        # dont un vrai. Le score 3 en tête est faux.
        scores = {('a','b'):3.0, ('a','c'):1.0, ('a','d'):1.0}
        test = {('a','c'), ('b','d')}
        rows, positifs = courbe_attendue(scores, test, 6, 6)
        self.assertEqual(positifs, 1)
        self.assertAlmostEqual(rows[0][1], 0)
        self.assertAlmostEqual(rows[1][1], .25)  # 0.5 bon en moyenne / 2
        self.assertAlmostEqual(rows[2][1], 1/3)
        self.assertAlmostEqual(rows[3][1], (1 + 1/3)/4)
        self.assertAlmostEqual(rows[-1][2], 1)
        self.assertEqual(top_k_deterministe(scores, test, 2, 6), 1)

    def test_scores_manquants_et_fin(self):
        rows, positif = courbe_attendue({}, {('a','b')}, 4, 4)
        self.assertEqual(positif, 0)
        self.assertAlmostEqual(rows[0][1], .25)
        self.assertAlmostEqual(rows[3][2], 1)
        self.assertIsNone(top_k_deterministe({}, {('a','b')}, 1, 4))


if __name__ == "__main__":
    unittest.main()
