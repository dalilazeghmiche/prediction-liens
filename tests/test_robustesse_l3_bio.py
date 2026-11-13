"""Tests de la comparaison multi-graines (sans telecharger HI-II-14)."""
import tempfile
import unittest
from pathlib import Path

from robustesse_l3_bio import (
    METHODES, deltas_apparies, executer, parser_graines, resume_par_methode
)


class TestRobustesseL3(unittest.TestCase):
    def test_graines(self):
        self.assertEqual(parser_graines("42,43,44,45,46"), [42,43,44,45,46])
        for bad in ("42", "42,42", "42,x", ""):
            with self.assertRaises(ValueError):
                parser_graines(bad)

    def test_resume_et_appariement(self):
        lignes = []
        for seed in (42, 43):
            for methode, p in (("CN", 2.0), ("AA", 3.0), ("L3_optimise", 6.0)):
                lignes.append(dict(seed=seed, methode=methode,
                                   precision_k_pourcent=p,
                                   rappel_positif_pourcent=10.0,
                                   temps_scores_s=0.1))
        resume = resume_par_methode(lignes)
        self.assertEqual(len(resume), 3)
        self.assertAlmostEqual(resume[2]["precision_moyenne_pct"], 6.0)
        self.assertAlmostEqual(resume[2]["precision_ecart_type_pct"], 0.0)
        deltas = deltas_apparies(lignes)
        self.assertEqual(len(deltas), 4)
        self.assertTrue(all(r["gain_points_precision"] > 0 for r in deltas))

    def test_avec_un_petit_reseau(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "hi.tsv"
            # Format Ensembl valide, y compris une boucle qui cree un isole.
            e = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,0),
                 (0,2),(1,3),(2,4),(3,5),(4,6),(5,0),(6,1),(7,7)]
            src.write_text(''.join(
                f'ENSG{1000+a:011d}\tENSG{1000+b:011d}\n' for a,b in e
            ), encoding="utf-8")
            out = Path(tmp) / "output"
            lignes, resume = executer(src, out, [42,43], .3, figures=False)
            self.assertEqual(len(lignes), 6)
            self.assertEqual(set(row["methode"] for row in lignes), set(METHODES))
            self.assertTrue(all(row["n_sommets"] == 8 for row in lignes))
            self.assertTrue(all(row["n_test"] == 4 for row in lignes))
            self.assertEqual(len(resume), 3)
            self.assertTrue((out / "resume_robustesse.csv").is_file())
            self.assertTrue((out / "protocole_robustesse.json").is_file())


if __name__ == "__main__":
    unittest.main()
