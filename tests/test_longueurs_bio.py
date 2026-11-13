"""Tests minimaux pour la comparaison A^2/A^3/A^4."""
import unittest
import numpy as np
from src.graph_io import from_edges
from src.measures.l3 import l3_optimise
from comparer_l2_l3_l4_bio import creer_matrice, histogramme_scores, courbe_depuis_groupes


class TestLongueurs(unittest.TestCase):
    def test_chaine_4_sommets(self):
        edges = {('X','U'), ('U','V'), ('V','Y')}
        nodes = sorted({'X','U','V','Y'})
        a, ids, neigh = creer_matrice(nodes, edges)
        b2, b3, b4 = a @ a, a @ a @ a, a @ a @ a @ a
        self.assertEqual(int(b2[ids['X'], ids['Y']]), 0)
        self.assertEqual(int(b3[ids['X'], ids['Y']]), 1)
        self.assertEqual(int(b4[ids['X'], ids['Y']]), 0)
        scores = l3_optimise(from_edges(edges, nodes=nodes))
        self.assertAlmostEqual(scores[('X','Y')], 0.5)

    def test_chaine_5_sommets(self):
        edges = {('A','B'), ('B','C'), ('C','D'), ('D','E')}
        nodes = sorted({'A','B','C','D','E'})
        a, ids, neigh = creer_matrice(nodes, edges)
        self.assertEqual(int((a @ a @ a)[ids['A'], ids['E']]), 0)
        self.assertEqual(int((a @ a @ a @ a)[ids['A'], ids['E']]), 1)

    def test_groupes_et_ex_aequo(self):
        edges = {('A','B'), ('B','C'), ('C','D'), ('D','E')}
        nodes = sorted({'A','B','C','D','E'})
        a, ids, neigh = creer_matrice(nodes, edges)
        m = a @ a @ a @ a
        # Parmi les candidats, la paire AE a un score strictement positif.
        test = [set() for _ in nodes]
        test[ids['A']].add(ids['E'])
        h, num_pos, hits_pos = histogramme_scores(m, neigh, test)
        curve = courbe_depuis_groupes(h, 10 - len(edges), 1, 6, (num_pos, hits_pos))
        self.assertEqual(curve[-1][0], 6)
        self.assertAlmostEqual(curve[-1][2], 1.)
        self.assertGreater(hits_pos, 0)


class TestEquivalenceCN(unittest.TestCase):
    def test_a2_est_cn_avec_un_graphe_aleatoire(self):
        from src.measures.local import cn_scores
        rng = np.random.default_rng(2026)
        for _ in range(15):
            nodes = list('ABCDEFGH')
            edges = {(nodes[i], nodes[j])
                     for i in range(len(nodes)) for j in range(i+1, len(nodes))
                     if rng.random() < .3}
            a, idx, adj = creer_matrice(nodes, edges)
            raw = a @ a
            graph = from_edges(edges, nodes=nodes)
            reference = cn_scores(graph)
            for i,x in enumerate(nodes):
                for y in nodes[i+1:]:
                    if y in graph[x]:
                        continue
                    self.assertEqual(int(raw[idx[x],idx[y]]), reference.get((x,y),0))

    def test_groupes_equivalents_evaluation_explicite(self):
        from evaluer_l3_bio import courbe_attendue
        nodes = list('ABCDEFG')
        edges = {('A','B'), ('B','C'), ('C','D'), ('D','E'), ('E','F'), ('B','F'), ('C','G')}
        test = {('A','F'),('A','C'),('C','E')}
        a, ids, neigh = creer_matrice(nodes, edges)
        test_nbr = [set() for _ in nodes]
        for x, y in test:
            test_nbr[ids[x]].add(ids[y])
        candidats = len(nodes)*(len(nodes)-1)//2-len(edges)
        for l in (2,3,4):
            m = a.copy()
            for _ in range(l-1):
                m = m @ a
            groups, num, hits = histogramme_scores(m, neigh, test_nbr)
            observed = courbe_depuis_groupes(groups, candidats, len(test), candidats, (num,hits))
            scores = {}
            for i,x in enumerate(nodes):
                for y in nodes[i+1:]:
                    if y in from_edges(edges, nodes=nodes)[x]:
                        continue
                    score = int(m[ids[x], ids[y]])
                    if score > 0:
                        scores[x,y] = score
            expected, expected_hits = courbe_attendue(scores, test, candidats, candidats)
            self.assertEqual(hits, expected_hits)
            self.assertEqual(len(observed), len(expected))
            for a_row,b_row in zip(observed,expected):
                self.assertEqual(a_row[0],b_row[0])
                self.assertAlmostEqual(a_row[1],b_row[1], places=12)
                self.assertAlmostEqual(a_row[2],b_row[2], places=12)


if __name__ == '__main__':
    unittest.main()
