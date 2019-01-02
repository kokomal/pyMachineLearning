# coding:utf-8
import math
import unittest
import sys
sys.path.append("../../")
from girvanNewman import communities


class GirvanNewmanTest(unittest.TestCase):
    def tearDown(self):
        print('***AFTER ONE TEST***')

    def setUp(self):
        print('***BEFORE ONE TEST***')

    @classmethod
    def tearDownClass(self):
         print('FINISHED...')

    @classmethod
    def setUpClass(self):
        print('STARTING...')
        
    def test_karate_club(self):
        ipt_txt = "../louvain/data/karate.txt"
        ipt_png = "image.png"
        c = communities.Communities(ipt_txt, ipt_png)
        c.initialize()
        c.display_graph()
        print("*"*70)
        partition, part_graph, removed_edges = c.find_best_partition()
        c.display(partition)
        c.plot_graph(part_graph, removed_edges)


if __name__ == '__main__':
    unittest.main()
