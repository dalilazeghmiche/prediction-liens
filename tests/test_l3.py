import unittest
from src.measures.l3 import l3_naif


class TestL3(unittest.TestCase):


    def test_naif_vs_optimise(self):
        from src.measures.l3 import l3_optimise

        graphe = {
            "X": {"A", "C"},
            "A": {"X", "B"},
            "B": {"A", "Y"},
            "C": {"X", "D"},
            "D": {"C", "Y"},
            "Y": {"B", "D"}
        }

        scores_naifs = l3_naif(graphe)
        scores_optimises = l3_optimise(graphe)

        self.assertEqual(
            set(scores_naifs),
            set(scores_optimises)
        )

        for paire in scores_naifs:
            self.assertAlmostEqual(
                scores_naifs[paire],
                scores_optimises[paire],
                places=12
            )

        # Deux chemins :
        # X-A-B-Y et X-C-D-Y
        # Chaque chemin contribue 0.5
        self.assertAlmostEqual(
            scores_optimises[("X", "Y")],
            1.0
        )

    def test_chaine_quatre_sommets(self):
        graphe = {
            "X": {"U"},
            "U": {"X", "V"},
            "V": {"U", "Y"},
            "Y": {"V"}
        }

        scores = l3_naif(graphe)

        # Chemin X-U-V-Y : 1 / sqrt(2 * 2)
        self.assertAlmostEqual(scores[("X", "Y")], 0.5)

        # Aucune arête existante ne doit être prédite.
        self.assertNotIn(("U", "V"), scores)

        # La paire X-V n'a pas de chemin de longueur 3.
        self.assertNotIn(("V", "X"), scores)


if __name__ == "__main__":
    unittest.main()