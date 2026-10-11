"""Tests de la lecture des graphes.

Lancer depuis la racine du projet :
    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph_io import (EdgeListFormatError, from_edges, has_edge,
                          num_edges, num_nodes, read_edge_list,
                          read_temporal_edges)


def write_tmp(contenu):
    """Écrit `contenu` dans un fichier temporaire et renvoie son chemin."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(contenu)
        return f.name


class TestFromEdges(unittest.TestCase):
    def test_from_edges(self):
        g = from_edges([("A", "B"), ("A", "C"), ("B", "C"), ("B", "D")])
        self.assertEqual(num_nodes(g), 4)
        self.assertEqual(num_edges(g), 4)
        self.assertTrue(has_edge(g, "A", "B"))
        self.assertTrue(has_edge(g, "B", "A"))      # non orienté
        self.assertFalse(has_edge(g, "A", "D"))
        self.assertFalse(has_edge(g, "A", "Z"))     # sommet inconnu

    def test_doublons_et_boucles(self):
        g = from_edges([("A", "B"), ("B", "A"), ("A", "A")])
        self.assertEqual(num_nodes(g), 2)
        self.assertEqual(num_edges(g), 1)

    def test_sommets_isoles(self):
        g = from_edges([("A", "B")], nodes=["A", "B", "C"])
        self.assertEqual(num_nodes(g), 3)
        self.assertEqual(g["C"], set())


class TestReadEdgeList(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "path"):
            os.remove(self.path)

    def test_lecture_simple(self):
        self.path = write_tmp("# commentaire\n0 1\n1\t2 1999\n\n2 0\n")
        g = read_edge_list(self.path, node_type=int)
        self.assertEqual(num_edges(g), 3)
        self.assertEqual(g[1], {0, 2})

    def test_ligne_trop_courte_strict(self):
        self.path = write_tmp("0 1\n2\n")
        with self.assertRaises(EdgeListFormatError) as ctx:
            read_edge_list(self.path, node_type=int)
        self.assertIn("ligne 2", str(ctx.exception))

    def test_ligne_trop_courte_tolerante(self):
        self.path = write_tmp("0 1\n2\n")
        g = read_edge_list(self.path, node_type=int, strict=False)
        self.assertEqual(num_edges(g), 1)

    def test_identifiant_non_convertible(self):
        self.path = write_tmp("0 1\n1 x\n")
        with self.assertRaises(EdgeListFormatError) as ctx:
            read_edge_list(self.path, node_type=int)
        self.assertIn("ligne 2", str(ctx.exception))


class TestReadTemporalEdges(unittest.TestCase):
    def tearDown(self):
        os.remove(self.path)

    def test_dates_lues(self):
        self.path = write_tmp("A B 1995\nA B 1998\nB C 1997\n")
        triples = read_temporal_edges(self.path)
        self.assertEqual(triples, [("A", "B", 1995), ("A", "B", 1998),
                                   ("B", "C", 1997)])

    def test_date_manquante(self):
        self.path = write_tmp("A B 1995\nA C\n")
        with self.assertRaises(EdgeListFormatError):
            read_temporal_edges(self.path)


if __name__ == "__main__":
    unittest.main()
