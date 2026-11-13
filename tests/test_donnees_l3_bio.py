"""Tests du chargement de HI-II-14 et du masquage sans fuite."""
import tempfile
import unittest
from pathlib import Path
from preparer_donnees_l3 import lire_hi_ii_14, preparer


class TestBio(unittest.TestCase):
    def test_nettoyage_et_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'mini.tsv'
            src.write_text(
                'protein1\tprotein2\n'
                'ENSG00000000001\tENSG00000000002\n'
                'ENSG00000000002\tENSG00000000001\n'
                'ENSG00000000002\tENSG00000000003\n'
                'ENSG00000000003\tENSG00000000004\n'
                'ENSG00000000001\tENSG00000000004\n'
                'ENSG00000000001\tENSG00000000001\n', encoding='utf-8')
            adj, stats = lire_hi_ii_14(src)
            self.assertEqual(stats['aretes_uniques'], 4)
            self.assertEqual(stats['doublons_ignores'], 1)
            self.assertEqual(stats['boucles_ignorees'], 1)
            r = preparer(src, Path(tmp) / 'sortie', .5, 42)
            self.assertEqual((r['aretes_train'], r['aretes_test']), (2, 2))
            self.assertEqual(r['paires_candidates'], 4)
            self.assertEqual(r['precision_hasard'], .5)
            self.assertTrue((Path(tmp)/'sortie'/'protocole.json').is_file())

    def test_fichier_non_valide(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'erreur.tsv'
            src.write_text('<html>\tpage</html>\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Ensembl'):
                lire_hi_ii_14(src)


if __name__ == '__main__':
    unittest.main()
