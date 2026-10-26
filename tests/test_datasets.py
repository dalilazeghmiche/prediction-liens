"""Tests du chargement de hep-th (sur une petite archive fabriquée).

Lancer depuis la racine du projet :
    python -m unittest discover -s tests -v
"""

import io
import os
import shutil
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets import (load_hepth, normalize_author, parse_abs,
                          split_authors)

ABS_1 = """------------------------------------------------------------------------------
\\\\
Paper: hep-th/9501001
From: Someone <x@y.z>
Date: Mon, 2 Jan 1995 14:42:25 +0200 (EET)   (24kb)

Title: Un titre
Authors: P.A. Grassi (SUNY, Stony Brook), G. Policastro (Pisa, Scuola Normale
  Superiore), and P. van Nieuwenhuizen (SUNY, Stony Brook)
Comments: 10 pages
\\\\
  Résumé...
\\\\
"""

ABS_2 = """\\\\
Paper: hep-th/9801002
Title: Autre titre
Author: Paolo Grassi
\\\\
"""


class TestNoms(unittest.TestCase):
    def test_split_authors(self):
        self.assertEqual(split_authors("A. Smith, B. Jones and C. Doe (MIT)"),
                         ["A. Smith", "B. Jones", "C. Doe"])
        self.assertEqual(split_authors("M. Alves, J. Salim, and K. Lee"),
                         ["M. Alves", "J. Salim", "K. Lee"])

    def test_normalisation(self):
        self.assertEqual(normalize_author("John Smith"), "j smith")
        self.assertEqual(normalize_author("J. Smith"), "j smith")
        self.assertEqual(normalize_author("J.Smith"), "j smith")
        self.assertEqual(normalize_author("John A. Smith Jr."), "j smith")
        self.assertEqual(normalize_author("Norma Manko\\v c"), "n mankoc")

    def test_parse_abs_multiligne_et_affiliations(self):
        self.assertEqual(parse_abs(ABS_1),
                         ["p grassi", "g policastro", "p nieuwenhuizen"])

    def test_parse_abs_un_auteur(self):
        self.assertEqual(parse_abs(ABS_2), ["p grassi"])


class TestLoadHepth(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.abs_tar = os.path.join(self.dir, "abs.tar.gz")
        self.dates_tar = os.path.join(self.dir, "dates.tar.gz")
        with tarfile.open(self.abs_tar, "w:gz") as tar:
            for name, txt in [("1995/9501001.abs", ABS_1),
                              ("1998/9801002.abs", ABS_2)]:
                data = txt.encode("latin-1")
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        with tarfile.open(self.dates_tar, "w:gz") as tar:
            # 9801002 a une date SLAC en 1997 : elle doit primer sur le dossier
            data = b"9501001 1995-01-02\n9801002 1997-12-30\n"
            info = tarfile.TarInfo("hep-th-slacdates")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_articles(self):
        papers = sorted(load_hepth(self.abs_tar, self.dates_tar))
        self.assertEqual(papers, [
            (1995, ["p grassi", "g policastro", "p nieuwenhuizen"]),
            (1997, ["p grassi"]),
        ])


if __name__ == "__main__":
    unittest.main()
    