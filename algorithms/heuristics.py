"""Two minimum-vertex-cut-based heuristic algorithms along with a naive greedy algorithm"""
from utils.core_solvers import solveMinVertexCut
from copy import copy
import numpy as np
import networkx as nx


def heuristicMinCut2(g, S, postProcess=True):
    """
    Min-cut based heuristic algorithm for minimum-cost intervention.
    The strategy is to break the ancestral sets where bidir(S) are present.
    :param g: an instance of ADMG
    :param S: set of nodes corresponding to the query Q[S]
    :param postProcess: if True, the optional heuristic post-process to reduce the cots.
    :return: a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
    """
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:    # dirP is enough for the identification of Q[S]
        return [dirP, g.interventionCost(dirP)]
    S_unid = set.union(*unid_comps)
    H = set.union(*[h.hedgeHull(c) for c in unid_comps])
    # Construct the graph \mathcal{H}:
    dirSubg = h.nodeSubgraph(H, directed=True)
    bidirS = h.bidir(S_unid, H)
    minCut = solveMinVertexCut(dirSubg, bidirS, S_unid, S)
    if minCut == 'inf':
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    if postProcess:
        minCut = heuristicPostProcess(h, S_unid, minCut)
    I = dirP.union(minCut)
    return [I, g.interventionCost(I)]


def heuristicMinCut1(g, S, postProcess=True):
    """
    Min-cut based heuristic algorithm for minimum-cost intervention.
    The strategy is to break the c-components where pa(S) are present.
    :param g: an instance of ADMG
    :param S: set of nodes corresponding to the query Q[S]
    :param postProcess: if True, the optional heuristic post-process to reduce the cots.
    :return: a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
    """
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # dirP is enough for the identification of Q[S]
        return [dirP, g.interventionCost(dirP)]
    S_unid = set.union(*unid_comps)
    H = set.union(*[h.hedgeHull(c) for c in unid_comps])
    # Construct the graph \mathcal{H}:
    biSubg = h.nodeSubgraph(H, directed=False)  # bidirected subgraph
    parentS = h.parents(S_unid, H)
    minCut = solveMinVertexCut(biSubg, parentS, S_unid, S)
    if minCut == 'inf':
        print('could not find a solution. returning infinite cost.')
        return [[], np.inf]
    if postProcess:
        minCut = heuristicPostProcess(h, S_unid, minCut)
    I = dirP.union(minCut)
    return [I, g.interventionCost(I)]


def naiveGreedy(g, S, postProcess=True):  # Reduce the sum of the cost of the remaining hedge hull
    """
    Naive greedy algorithm for minimum cost intervention.
    Intervenes upon nodes greedily until Q[S] becomes identifiable.
    :param g: an ADMG g
    :param S: set of nodes corresponding to the query Q[S]
    :param postProcess: if True, the optional heuristic post-process to reduce the cots.
    :return: a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
    """
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    dirP = []
    for comp in comps:
        dirP += list(g.directParents(comp))
    dirP = set(dirP)
    h = copy(g)
    h.permIntervene(dirP)  # permanently intervene on direct parents. We need them any way
    unid_comps = [c for c in comps if not h.isIdentifiable(c)]
    if len(unid_comps) == 0:  # identifiability already achieved.
        return [dirP, g.interventionCost(dirP), 0]
    H = {tuple(c): h.hedgeHull(c) for c in unid_comps}  # hedge hulls of each component
    H_uni = set.union(*H.values())
    appearances = {u: [c for c in unid_comps if u in H[tuple(c)]] for u in H_uni.difference(S)}  # which node
    I = []
    identified_comp = {tuple(c): False for c in unid_comps}
    while True:
        greedyGain = {u: len(appearances[u]) / g.interventionCost({u}) for u in appearances.keys()}  # gain of
        xtoIntervene = max(greedyGain, key=greedyGain.get)
        I.append(xtoIntervene)
        for c in appearances[xtoIntervene]:
            if not identified_comp[tuple(c)]:
                new_hh = h.hedgeHull(c, H[tuple(c)].difference([xtoIntervene]))
                for v in H[tuple(c)].difference(new_hh).difference(S):
                    appearances[v].remove(c)
                if new_hh == c:
                    identified_comp[tuple(c)] = True
                else:
                    H[tuple(c)] = new_hh
        if all(identified_comp.values()):
            break
    if postProcess:
        I = heuristicPostProcess(h, S, I)
    I = dirP.union(I)
    return [I, g.interventionCost(I)]


def heuristicPostProcess(g, S, A):
    """
    a heuristic post-process to reduce the cost of heuristic algorithms.
    make A smaller as long as Q[S] is identifiable
    :param g: an ADMG
    :param S: set of nodes corresponding to the query Q[S]
    :param A: a list containing the output of a heuristic alg.
    :return: a set which is sufficient to intervene upon for identification of Q[S], along with its cost.
    """
    weights = g.get_nodeCosts()
    comps = nx.connected_components(g.nodeSubgraph(S, directed=False))  # C-components of S
    comps = [c for c in comps]
    A, _ = (list(x) for x in zip(*sorted({a: weights[a] for a in A}.items(), key=lambda item: item[1])))
    V = list(set(g.nodes).difference(A))
    for a in A:
        if all(g.isIdentifiable(c, set(V+[a])) for c in comps):
            V += [a]
    return set(g.nodes).difference(V)