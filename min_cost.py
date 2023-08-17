import networkx as nx
from admg.admg import ADMG
from algorithms.main_alg import Alg2
from algorithms.heuristics import heuristicMinCut2, heuristicMinCut1, naiveGreedy


if __name__ == '__main__':

    # build a DiGraph and a Graph:
    g_dir = nx.DiGraph()  # the directed component of the ADMG
    g_bi = nx.Graph()  # the bidirected component of the ADMG
    nodes = [('a', {'weight': 5}), ('2', {'weight': 4}), ('3', {'weight': 3}), ('4', {'weight': 5}),
             ('5', {'weight': 1})]
    g_dir.add_nodes_from(nodes)
    g_bi.add_nodes_from(nodes)
    dir_edges = [('a', '2'), ('3', '2'), ('4', 'a'),
                 ('5', '3'), ('4', '5')]
    bi_edges = [('a', '4'), ('5', '4'), ('4', '2'),
                ('4', '3')]
    g_dir.add_edges_from(dir_edges)
    g_bi.add_edges_from(bi_edges)

    # build an instance of ADMG with g_dir and g_bi:
    g = ADMG(g_dir, g_bi)
    Y = {'2'}

    intervention, cost = heuristicMinCut2(g, Y)   # heuristic min-cut based algorithm 2 for min-cost intervention design
    print('heuristicMinCut2: [' + str(intervention) + ', ' + str(cost) + ']')

    intervention, cost = heuristicMinCut1(g, Y)        # heuristic min-cut based algorithm 1 for min-cost
    print('heuristicMinCut1: [' + str(intervention) + ', ' + str(cost) + ']')

    intervention, cost = naiveGreedy(g, Y, postProcess=True)            # a naive greedy for min-cost intervention
    print('Naive Greedy: [' + str(intervention) + ', ' + str(cost) + ']')

    intervention, cost = g.optimalIntervention(Y)  # brute-force algorithm for min-cost intervention which checks
    # for every subset
    print('Brute force: [' + str(intervention) + ', ' + str(cost) + ']')

    intervention, cost, _ = Alg2(g, Y)              # minimum-hitting set based more efficient algorithm for
    # min-cost intervention
    print('Exact Alg2: [' + str(intervention) + ', ' + str(cost) + ']')

    intervention, cost, _ = Alg2(g, Y, hittingSetExactSolver=False)     # same algorithm, but solves the
    # min-hitting set approximately
    print('Approx. Alg2: [' + str(intervention) + ', ' + str(cost) + ']')
